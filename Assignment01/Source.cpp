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

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/filesystem.hpp>

using namespace cv;
using namespace std;


/* A bunch of constants
*/
//todo: maybe not make them global and move them into main?
const string outputFile = "calib.xml";
// Path to the images folder
const filesystem::path IMAGES_PATH = filesystem::absolute("../imgs/02/");
// Size of the chessboard, measured from the inner corners
const Size BOARDSIZE = Size(9, 6);
// Square size in mm
constexpr int SQUARESIZE = 24;
// Width of the chess grid in mm
constexpr int GRID_WIDTH = 226; //SQUARESIZE * 9;

// Amount of succesful corner checks we want before we calibrate
constexpr int MAXDETECTIONS = 15;
// Webcam delay between snapshots for corner checking
constexpr int DELAY = 100;
// The aspect ratio of your camera
constexpr float ASPECT_RATIO = 4 / 3;

// Flags for checking for chessboard corners, finetuning can be done here
constexpr int CHESSFLAGS = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
// Flags for the calibration step. I have no idea what most of these do.
// TODO: Figure out how to improve stuff by changing some flags around.
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
		namedWindow("First CV Assignment", WINDOW_AUTOSIZE);
		moveWindow("First CV Assignment", 0, 45);
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
	void GatherData(int& no_squares)
	{
		
		// Iterates every image in the given directory and gathers the chess corner points 
		if (capture_mode == IMAGEFOLDER)
		{
			for (const String image_path : images.imageList)
			{
				img = imread(image_path);
				imageSize = img.size();

				// The photos I took with my phone are 4000 x 3000, which makes finding the corners reeaallly slow, so we resize it
				//resize(img, img, Size(1600, 900));

				// Grayscale each image, color adds nothing
				Mat gray;
				cvtColor(img, gray, COLOR_BGR2GRAY);

				// Find the corners
				vector<Point2f> corners;
				bool ret = findChessboardCorners(gray, BOARDSIZE, corners, CHESSFLAGS);
				if (ret)
				{
					cout << std::format("Chessboard corners found on image: {}\n", image_path);

					// Improve accuracy on corners with some subpixel magic
					cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
					pointMatrix.push_back(corners);

					// Draw and show the corners for funsies
					drawChessboardCorners(img, BOARDSIZE, corners, ret);
					imshow("First CV Assignment", img);

					// Wait for keypress
					waitKey(0);
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

					// Grayscale the image
					cvtColor(img, img, COLOR_BGR2GRAY);

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
		destroyWindow("First CV Assignment");
	}

	/* Calibrate using the corner data gathered with GatherData()
	*/
	//bool CalibrateCamera()
	void CalibrateCamera(double& rms, Mat& cameraMatrix, Mat& distCoeffs, Mat& rvecs, Mat& tvecs)
	{
		vector<vector<Point3f>> objectPoints(1);
		vector<Point3f> newObjPoints;

		CalculateCornerPositions(BOARDSIZE, SQUARESIZE, objectPoints[0]);
		// Do some math-y math for use case where our piece of paper is an imperfect planar target
		objectPoints[0][BOARDSIZE.width - 1].x = objectPoints[0][0].x + GRID_WIDTH;
		newObjPoints = objectPoints[0];
		objectPoints.resize(pointMatrix.size(), objectPoints[0]);

		// Ready some parameters
		const int iFixedPoint = BOARDSIZE.width - 1;
		cameraMatrix = Mat::eye(3, 3, CV_64F);
		distCoeffs = Mat::zeros(8, 1, CV_64F);

		// Calibration time!
		//iFixedPoint = -1;
		 rms = calibrateCameraRO(objectPoints, pointMatrix, imageSize, iFixedPoint, cameraMatrix, distCoeffs,
		                               rvecs, tvecs, newObjPoints, CALIB_USE_LU); //CALIB_USE_LU faster, less acc

		// Tell us the overall RMS error
		cout << "Calibration overall RMS re-projection error:\t" << rms << "\n";
		// Check if everything went correctly
		bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);
		
		cout << "Camera Matrix:\n" << cameraMatrix << "\nDistortion Coefficients:\n" << distCoeffs << "\n";

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

	// Calculate the object points of the chessboard
	static void CalculateCornerPositions(const Size boardSize, const int squareSize, vector<Point3f>& out) 
	{
		for (int i = 0; i < boardSize.height; i++)
		{
			for (int j = 0; j < boardSize.width; j++)
			{
				// Depth(Z) of 0 for chessboard!
				out.push_back(Point3f(j * squareSize, i * squareSize, 0));
			}
		}
	}

	/* This function loops, showing the webcam feed.
	* Every so often, depending on the delay variable, it copies a new image for use in calibration.
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

private:
	mode capture_mode;
	Mat img;
	ImageReader images;
	atomic<bool> imageReady = false;
	mutex img_mutex;
	clock_t prevTimestamp = 0;
	Size imageSize;
	//thread t1;
};


/* Note: The squares on the paper seem to be 22mm wide and long */
int main(int argc, char* argv[])
{

	//vars
	Mat cameraMatrix, distCoeffs, rvecs, tvecs; 
	double rms = 0.0;

	int no_squares = 0;
	// Using images
	Calibrator calibrator = Calibrator{IMAGES_PATH};

	// Using a webcam
	//Calibrator calibrator = Calibrator(0);

	calibrator.GatherData(no_squares);
	cout << "Data gathered successfully. Maybe. Hopefully.\n";
	cout << "number of images omited:" << no_squares << "\n";

	calibrator.CalibrateCamera(rms, cameraMatrix,distCoeffs, rvecs, tvecs);
	Save_calibration(cameraMatrix, distCoeffs, rms, rvecs, tvecs);

	//Click to exit
	waitKey(0);
}


