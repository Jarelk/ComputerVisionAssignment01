#include <format>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <utility>
#include <vector>
#include <thread>
#include <math.h>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/filesystem.hpp>

//fancy colors BGR
#define BLUE Scalar(255,0,0)
#define GREEN Scalar(0,255,0)
#define RED Scalar(0,0,255)
#define YELLOW Scalar(0,255,255) 

using namespace cv;
using namespace std;


/* A bunch of constants
*/
const string outputFile = "calib.xml";
// Path to the images folder
const filesystem::path IMAGES_PATH = filesystem::absolute("../imgs/02/");
// Size of the chessboard, measured from the inner corners
const Size BOARDSIZE = Size(9, 6);
// Square size in mm
//constexpr int SQUARESIZE = 22;
constexpr int SQUARESIZE = 24;
// Width of the chess grid in mm
//constexpr int GRID_WIDTH = 176; //SQUARESIZE * 9;
constexpr int GRID_WIDTH = 226;

// Amount of succesful corner checks we want before we calibrate
constexpr int MAXDETECTIONS = 15;
// Webcam delay between snapshots for corner checking
constexpr int DELAY = 100;
// The aspect ratio of your camera
constexpr float ASPECT_RATIO = 4 / 3;

// Flags for checking for chessboard corners, finetuning can be done here
constexpr int CHESSFLAGS = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
// Flags for the calibration step. I have no idea what most of these do.
constexpr int CALIBFLAG = 0;

class ImageReader 
{
public:
	// Just use the filesystem iterator to find the images
	vector<String> imageList;

	ImageReader() = default;

	/* Constructor needs directory path
	*/
	ImageReader(const filesystem::path& path)
	{
		// Read the directory and globs the found jpg's in a list
		utils::fs::glob(path.string(), "*.jpg", imageList, false);
	}
};

void Save_calibration(const Mat& cameraMatrix, const Mat& distCoeffs, double& rms, Mat& rvecs, Mat& tvecs)
{

	FileStorage fs(outputFile, FileStorage::WRITE);
	//write calibration data:
	fs << "camera_matrix" << cameraMatrix;
	fs << "distortion_coefficients" << distCoeffs;
	fs << "avg_reprojection_error" << rms;
	//extrinsics params ?
	fs << "rvecs" << rvecs;
	fs << "tvecs" << tvecs;
};

/* TODO: Remove duplicate code for data gathering
*/
class Calibrator
{
public:
	enum mode {WEBCAM, IMAGEFOLDER};
	vector<vector<Point2f>> pointMatrix;

	Calibrator() 
	{
		//namedWindow("First CV Assignment", WINDOW_AUTOSIZE);
		//moveWindow("First CV Assignment", 0, 45);
	}

	/* Constructor for when you're using an image path
	* PARAMS:
	*	fileystem::path path: The (absolute) path to the image folder
	*/
	Calibrator(filesystem::path path) : Calibrator()
	{
		images = ImageReader(std::move(path));
		capture_mode = IMAGEFOLDER;
	}

	/* Constructor for when you use a camera
	* PARAMS:
	*	int cameraID: The id of the camera (for webcams this is usually 0)
	*/
	Calibrator(int cameraID) : Calibrator()
	{
		VideoCapture webcam(0);

		// If the webcam doesn't open something might be wrong
		if (!webcam.isOpened()) {
			throw runtime_error("Ey yo ma the webcam broke?");
		}
		capture_mode = WEBCAM;

		// Open a window for our webcam
		namedWindow("Stream", WINDOW_AUTOSIZE);
		moveWindow("Stream", 0, 45);

		// Time for a stream
		//t1 = thread(&Calibrator::DisplayCamera, this, ref(webcam)); //WEBCAM ERROR HERE

	}

	/* Gathers data by iterating over webcam images or by reading images from a folder */
	void GatherData(int& no_squares, vector<Point2f>& corners)
	{
		
		// Iterates every image in the given directory and gathers the chess corner points 
		if (capture_mode == IMAGEFOLDER)
		{
			
			for (const String image_path : images.imageList)
			//for (const String image_path : final_img_list) //with best imgs only
			{
				img = imread(image_path);
				imageSize = img.size();

				// The photos I took with my phone are 4000 x 3000, which makes finding the corners reeaallly slow, so we resize it
				//resize(img, img, Size(1600, 900));

				// Find the corners //todo: use this for online phase too
				//vector<Point2f> corners; ->moved to int
				bool ret = findChessboardCorners(img, BOARDSIZE, corners, CHESSFLAGS);
				if (ret)
				{
					cout << std::format("Chessboard corners found on image: {}\n", image_path);

					//Grayscale image for subpixel
					Mat gray;
					cvtColor(img, gray, COLOR_BGR2GRAY);

					// Improve accuracy on corners with some subpixel magic
					cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
					pointMatrix.push_back(corners);

					// Draw and show the corners for funsies
					//drawChessboardCorners(img, BOARDSIZE, corners, ret);
					
					//imshow("First CV Assignment", img);

					// Wait for keypress
					//waitKey(0);
				}
				else
				{
					cout << std::format("Chessboard corner search unsuccessful for image: {}\n", image_path);
					no_squares++;
				}

			}
		}
		// Takes webcam snapshots and finds the corners using that
		if (capture_mode == WEBCAM) 
		{
			// The amount of succesful corner detections we want to calibrate
			int iterator = 0;
			while (iterator < MAXDETECTIONS) {
				// If we have an image to compute with
				if (imageReady) {
					//Lock the mutex
					img_mutex.lock();
					imageSize = img.size();

					// Find the corners
					vector<Point2f> corners;
					bool ret = findChessboardCorners(img, BOARDSIZE, corners, CHESSFLAGS);

					// If we found corners:
					if (ret) 
					{
						// Let the audience know we succesfully found corners
						cout << "Corners found on snapshot.\n";
						
						//+1 on the iterator!
						iterator++;

						//Grayscale image for subpixel
						Mat gray;
						cvtColor(img, gray, COLOR_BGR2GRAY);

						// Improve accuracy on corners with some subpixel magic
						cornerSubPix(img, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));

						// Push the found corners on the matrix
						pointMatrix.push_back(corners);

						// Draw and show the corners for funsies
						drawChessboardCorners(img, BOARDSIZE, corners, ret);
						imshow("First CV Assignment", img);

						// I remember from working with OpenCV in python that this is a real big neccessary line whenever you use imshow
						waitKey(1);
					}
					else 
					{
						cout << "Corner search unsuccessful on snapshot.\n";
					}

					// Unlock the mutex
					img_mutex.unlock();

					// Flag that we need a new image
					imageReady = false;
				}
			}
		}
		// Destroy the window
		//destroyWindow("First CV Assignment");
	}

	/* Calibrate using the corner data gathered with GatherData()
	*/
	
	void CalibrateCamera(double& rms, Mat& cameraMatrix, Mat& distCoeffs, Mat& rvecs, Mat& tvecs, int& no_squares, vector<Point2f>& corners)
	{
		vector<vector<Point3f>> objectPoints(1);
		vector<Point3f> newObjPoints;
		vector<double> perViewErrors;

		CalculateCornerPositions(objectPoints[0]);

		//Do some math-y math for use case where our piece of paper is an imperfect planar target
		objectPoints[0][BOARDSIZE.width - 1].x = objectPoints[0][0].x + GRID_WIDTH;
		newObjPoints = objectPoints[0];
		objectPoints.resize(pointMatrix.size(), objectPoints[0]);

		// Ready some parameters
		const int iFixedPoint = BOARDSIZE.width - 1;

		// Calibration time!
		Mat std1, std2, std3;
		rms = calibrateCameraRO(objectPoints, pointMatrix, imageSize, iFixedPoint, cameraMatrix, distCoeffs,
			rvecs, tvecs, newObjPoints, std1, std2, std3, perViewErrors, CALIB_USE_LU); //CALIB_USE_LU faster, less acc

		cout << std::format("Average rms: {}\n", rms);
		for (int i = 0; i < pointMatrix.size(); i++) 
		{
			if (perViewErrors.at(i) > rms) {
				cout << std::format("Removing image {} with rms {} \n", i, perViewErrors.at(i));
				pointMatrix.at(i) = vector<Point2f>(0);
			}
		}
		auto it = remove_if(pointMatrix.begin(), pointMatrix.end(), [](vector<Point2f> vec) {return vec == vector<Point2f>(0); });
		pointMatrix.erase(it, pointMatrix.end());

		// Repeat all the earlier math, with the new image subset
		vector<vector<Point3f>> objectPoints2(1);
		vector<Point3f> newObjPoints2;
		CalculateCornerPositions(objectPoints2[0]);
		objectPoints2[0][BOARDSIZE.width - 1].x = objectPoints2[0][0].x + GRID_WIDTH;
		newObjPoints2 = objectPoints2[0];
		objectPoints2.resize(pointMatrix.size(), objectPoints2[0]);

		// Calibration time! Again!
		rms = calibrateCameraRO(objectPoints2, pointMatrix, imageSize, iFixedPoint, cameraMatrix, distCoeffs,
			rvecs, tvecs, newObjPoints2, std1, std2, std3, perViewErrors, CALIB_USE_LU); //CALIB_USE_LU faster, less acc

		// Tell us the overall RMS error
		cout << "Calibration overall RMS re-projection error:\t" << rms << "\n";
		// Check if everything went correctly
		bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);
		
		cout << "Camera Matrix:\n" << cameraMatrix << "\nDistortion Coefficients:\n" << distCoeffs << "\n";

	}
	
	/* Shows the undistorted image in a window
	* PARAMS:
	*	Mat cameraMatrix:	the intrinsic camera parameters
	*	Mat distCoeffs:		the distortion coefficients
	*/
	void ShowUndistortedImages(Mat& cameraMatrix, Mat& distCoeffs)
	{
		if (capture_mode == IMAGEFOLDER) {
			cout << "Showing undistorted image.\n";
			Mat rview, map1, map2;
			initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
				getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
				imageSize, CV_16SC2, map1, map2);
			namedWindow("Undistorted", WINDOW_AUTOSIZE);
			moveWindow("Undistorted", 0, 45);
			for (const String image_path : images.imageList)
			{
				img = imread(image_path);
				remap(img, rview, map1, map2, INTER_LINEAR);

				imshow("Undistorted", rview);

				// Wait for keypress
				waitKey(0);
			}
		}
	}

	/* Calculate the object points of the chessboard
	* PARAMS:
	*	vector<Point3f>& out:	rvalue of the vector to fill with the corner data
	*/
	static void CalculateCornerPositions( vector<Point3f>& out) 
	{
		for (int i = 0; i < BOARDSIZE.height; i++)
		{
			for (int j = 0; j < BOARDSIZE.width; j++)
			{
				// Depth(Z) of 0 for chessboard!
				out.push_back(Point3f(j * SQUARESIZE, i * SQUARESIZE, 0));
			}
		}
	}

	/* This function loops, showing the webcam feed.
	* Every so often, depending on the delay variable, it copies a new image for use in calibration.
	* CURRENTLY NOT WORKING
	* */
	void DisplayCamera(VideoCapture& webcam) {
		while (true) {
			Mat stream; 
			webcam >> stream;
			imshow("Stream", stream);//EXCEPTION THROWN WHEN TRYING TO LOAD WEBCAM
			waitKey(1);

			if (clock() - prevTimestamp > DELAY && !imageReady) {
				prevTimestamp = clock();
				img_mutex.lock();
				stream.copyTo(img);
				true >> imageReady;
				img_mutex.unlock();
			}
		}
	}

public:
	mode capture_mode;
	Mat img;
	ImageReader images;
	atomic<bool> imageReady = false;
	mutex img_mutex;
	clock_t prevTimestamp = 0;
	Size imageSize;
	//thread t1;
};

/* Draws in a live loop on a chessboard using a webcam stream
* PARAMS:
*	Mat cameraMatrix: the intrinsic camera parameters found
*	Mat distCoeffs: the distortion coefficients found
*/
void draw_on_webcam(const Mat& cameraMatrix, const Mat& distCoeffs)
{
	//VideoCapture webcam("http://192.168.1.144:4747/mjpegfeed");
	VideoCapture webcam(0);

	if (!webcam.isOpened()) {
		cerr << "Webcam not opened!";
		throw runtime_error("Problem encountered while opening webcam stream.");
	}

	Mat stream;
	const string x = "x"; const string y = "y"; const string z = "-z";

	vector<Point3f> out; // objectPoints
	Calibrator::CalculateCornerPositions(out);
	constexpr int cub= SQUARESIZE * 2;

	vector<Point2d> point2D;//to draw
	vector<Point3d> point3D; //to draw

	//end points for axis lines
	point3D.push_back(Point3d(90.0, 0, 0));	//end point x
	point3D.push_back(Point3d(0, 90.0, 0));	//endpoint y
	point3D.push_back(Point3d(0, 0, -90.0));//end point z
	//vertices points for cube drawing
	point3D.push_back(Point3d(0, 0, -cub));	//vertix e
	point3D.push_back(Point3d(0, cub, -cub));	//vertix f
	point3D.push_back(Point3d(cub, 0, -cub));	//verix g 
	point3D.push_back(Point3d(cub, cub, -cub));	//vertix h 

	clock_t t;
	vector<Point3d> balls;
	vector<float> ZBuffer;
	Mat rvec, tvec;
	vector<Point2f> corners;
	balls.push_back(Point3d(100, 75, 0));
	balls.push_back(Point3d(120, 75, 0));
	ZBuffer.push_back(0.0);
	ZBuffer.push_back(0.0);
		while (true)
		{
			t = clock();
			float time = ((float)t / 500.0f);
			balls[0].z = - sinf(fmod(time, (float)CV_PI)) * SQUARESIZE * 2;
			balls[1].z = - sinf(fmod(time + 5, (float)CV_PI)) * SQUARESIZE * 3;

			webcam >> stream;
			if (bool has_corners = findChessboardCorners(stream, BOARDSIZE, corners))
			{
				//pose stimation. Orientation 3d in 2d img
				solvePnP(out, corners, cameraMatrix, distCoeffs, rvec, tvec);

				// Find Cameraposition
				Mat rMat;
				Rodrigues(rvec, rMat);
				Mat camMat = -1 * rMat.t() * Mat(tvec);
				Point3d camPos = Point3d(camMat);

				// Fill the "Z Buffer"
				ZBuffer[0] = norm(camPos - balls[0]);
				ZBuffer[1] = norm(camPos- balls[1]);


				//Projects 3D points to an image plane. 
				projectPoints(point3D, rvec, tvec, cameraMatrix, distCoeffs, point2D);

				//Project ball?
				vector<Point2d> ballimg;
				projectPoints(balls, rvec, tvec, cameraMatrix, distCoeffs, ballimg);
				
				//---------draw axis---------//todo: move out
				arrowedLine(stream, corners[0], point2D[0], BLUE,3);
				putText(stream, x, Point(point2D[0].x + 20, point2D[0].y), FONT_HERSHEY_SIMPLEX, 1, BLUE, 2);
				
				//y
				arrowedLine(stream, corners[0], point2D[1], GREEN, 3);
				putText(stream, y, Point(point2D[1].x - 10, point2D[1].y - 10), FONT_HERSHEY_SIMPLEX, 1, GREEN, 2);
				//z
				// arrowedLine(image, start_point, end_point, color, thickness)
				arrowedLine(stream, corners[0], point2D[2], RED, 3);
				putText(stream, z, Point(point2D[2].x - 10, point2D[2].y - 10), FONT_HERSHEY_SIMPLEX, 1, RED, 2);

				
				//drawFrameAxes(stream, cameraMatrix, distCoeffs, rvec, t, 30, 3); //draw axis
				
				//-----------draw cube-----------////TODO: DO THIS BETTER THANKS
							//letter belong to vertices. See reference here https://i.ibb.co/cvBScHW/cube-ref.png
				line(stream, corners[0], point2D[3], YELLOW, 2);	//a-e
				line(stream, point2D[5], corners[2], YELLOW, 2);	//g-d
				line(stream, point2D[4], corners[18], YELLOW, 2);	//f-b
				line(stream, point2D[6], corners[20], YELLOW, 2);	//h-c
				line(stream, corners[20], corners[2], YELLOW, 2);	//c-d
				line(stream, corners[0], corners[2], YELLOW, 2);	//a-d
				line(stream, corners[18], corners[0], YELLOW, 2);	//b-a
				line(stream, corners[18], corners[20], YELLOW, 2);	//b-c
				line(stream, point2D[6], point2D[5], YELLOW, 2);	//h-g
				line(stream, point2D[4], point2D[3], YELLOW, 2);	//f-e
				line(stream, point2D[3], point2D[5], YELLOW, 2);	//e-g
				line(stream, point2D[4], point2D[6], YELLOW, 2);	//f-h
				
				// If red is further than blue, draw red first
				if (ZBuffer[0] > ZBuffer[1]) {
					//cout << "Drawing red, then blue\n";
					circle(stream, ballimg[0], 20, RED, FILLED);
					circle(stream, ballimg[1], 20, BLUE, FILLED);
				}
				// If red is not further than blue, draw blue first
				else {
					//cout << "Drawing blue, then red\n";
					circle(stream, ballimg[1], 20, BLUE, FILLED);
					circle(stream, ballimg[0], 20, RED, FILLED);
				}
				
				
			}
			
			imshow("Assignment1", stream);
			waitKey(1);
		}

}

int main(int argc, char* argv[])
{
	//vars
	Mat cameraMatrix, distCoeffs, rvecs, tvecs;
	cameraMatrix = Mat::eye(3, 3, CV_64F);
	distCoeffs = Mat::zeros(8, 1, CV_64F);

	ImageReader img;

	double rms = 0.0;
	vector<Point2f> corners;

	int no_squares = 0;
	// Using images
	
	Calibrator calibrator = Calibrator{IMAGES_PATH};
	
	// Using a webcam
	//Calibrator calibrator = Calibrator(0);
	
	calibrator.GatherData(no_squares, corners);
	
	cout << "Data gathered successfully. Maybe. Hopefully.\n";
	cout << "number of images omited:" << no_squares << "\n";
	calibrator.CalibrateCamera(rms, cameraMatrix,distCoeffs, rvecs, tvecs, no_squares, corners);
	Save_calibration(cameraMatrix, distCoeffs, rms, rvecs, tvecs);

	//calibrator.ShowUndistortedImages(cameraMatrix, distCoeffs); todo: ANA UNCOMMENT
	draw_on_webcam(cameraMatrix, distCoeffs);

	//Click to exit
	waitKey(0);
}


