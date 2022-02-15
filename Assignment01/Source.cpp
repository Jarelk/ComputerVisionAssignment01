#include <format>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
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

/* A bunch of constants
*/
// Path to the images folder
const std::filesystem::path IMAGES_PATH = std::filesystem::absolute("../imgs/");
// Size of the chessboard, measured from the inner corners
const cv::Size BOARDSIZE = cv::Size(9, 6);
// Square size in mm
const int SQUARESIZE = 22;
// Width of the chess grid in mm
const int GRID_WIDTH = 198;

// Amount of succesful corner checks we want before we calibrate
const int MAXDETECTIONS = 15;
// Webcam delay between snapshots for corner checking
const int DELAY = 100;
// The aspect ratio of your camera
const float ASPECT_RATIO = 4 / 3;

// Flags for checking for chessboard corners, finetuning can be done here
const int CHESSFLAGS = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
// Flags for the calibration step. I have no idea what most of these do.
// TODO: Figure out how to improve stuff by changing some flags around.
const int CALIBFLAG = CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_PRINCIPAL_POINT | CALIB_FIX_ASPECT_RATIO | CALIB_ZERO_TANGENT_DIST;

class ImageReader
{
public:
	// Just use the filesystem iterator to find the images
	std::vector<cv::String> imageList;

	ImageReader() = default;

	/* Constructor needs directory path
	*/
	ImageReader(std::filesystem::path path)
	{
		// Read the directory and globs the found jpg's in a list
		cv::utils::fs::glob(path.string(), "*.jpg", imageList, false);
	}
};

/* TODO: Remove duplicate code for data gathering
*/
class Calibrator
{
public:
	enum mode {WEBCAM, IMAGEFOLDER};
	std::vector<std::vector<Point2f>> pointMatrix;

	Calibrator() 
	{
		namedWindow("First CV Assignment", WINDOW_AUTOSIZE);
		cv::moveWindow("First CV Assignment", 0, 45);
	}

	/* Constructor for when you're using an image path
	* PARAMS:
	*	std::fileystem::path path: The (absolute) path to the image folder
	*/
	Calibrator(std::filesystem::path path) : Calibrator()
	{
		images = ImageReader(path);
		capture_mode = IMAGEFOLDER;
	}

	/* Constructor for when you use a camera
	* PARAMS:
	*	int cameraID: The id of the camera (for webcams this is usually 0)
	*/
	Calibrator(int cameraID) : Calibrator()
	{
		cv::VideoCapture webcam(cameraID);

		// If the webcam doesn't open something might be wrong
		if (!webcam.isOpened()) {
			throw std::runtime_error("Ey yo ma the webcam broke?");
		}
		capture_mode = WEBCAM;

		// Open a window for our webcam
		namedWindow("Stream", WINDOW_AUTOSIZE);
		cv::moveWindow("Stream", 0, 45);

		// Time for a stream
		t1 = std::thread(&Calibrator::DisplayCamera, this, std::ref(webcam));

	}

	/* Gathers data by iterating over webcam images or by reading images from a folder */
	void GatherData()
	{
		// Iterates every image in the given directory and gathers the chess corner points 
		if (capture_mode == IMAGEFOLDER)
		{
			for (const cv::String image_path : images.imageList)
			{
				img = cv::imread(image_path);
				imageSize = img.size();

				// The photos I took with my phone are 4000 x 3000, which makes finding the corners reeaallly slow, so we resize it
				cv::resize(img, img, Size(1600, 900));

				// Grayscale each image, color adds nothing
				cv::Mat gray;
				cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

				// Find the corners
				std::vector<Point2f> corners;
				bool ret = cv::findChessboardCorners(gray, BOARDSIZE, corners, CHESSFLAGS);
				if (ret)
				{
					std::cout << std::format("Chessboard corners found on image: {}\n", image_path);

					// Improve accuracy on corners with some subpixel magic
					cv::cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
					pointMatrix.push_back(corners);

					// Draw and show the corners for funsies
					cv::drawChessboardCorners(img, BOARDSIZE, corners, ret);
					imshow("First CV Assignment", img);

					// Wait for keypress
					cv::waitKey(0);
				}
				else
				{
					std::cout << std::format("Chesboard corner search unsuccesful for image: {}\n", image_path);
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
					cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

					// Find the corners
					std::vector<Point2f> corners;
					bool ret = cv::findChessboardCorners(img, cv::Size{ 9, 6 }, corners, CHESSFLAGS);

					// If we found corners:
					if (ret) 
					{
						// Let the audience know we succesfully found corners
						std::cout << "Corners found on snapshot.\n";
						
						//+1 on the iterator!
						iterator++;

						// Improve accuracy on corners with some subpixel magic
						cv::cornerSubPix(img, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));

						// Push the found corners on the matrix
						pointMatrix.push_back(corners);

						// Draw and show the corners for funsies
						cv::drawChessboardCorners(img, Size{ 9, 6 }, corners, ret);
						imshow("First CV Assignment", img);

						// I remember from working with OpenCV in python that this is a real big neccessary line whenever you use cv::imshow
						cv::waitKey(1);
					}
					else 
					{
						std::cout << "Corner search unsuccesful on snapshot.\n";
					}

					// Unlock the mutex
					img_mutex.unlock();

					// Flag that we need a new image
					imageReady = false;
				}
			}
		}
		// Destroy the window
		cv::destroyWindow("First CV Assignment");
	}

	/* Calibrate using the corner data gathered with GatherData()
	*/
	bool CalibrateCamera() 
	{
		std::vector<std::vector<Point3f>> objectPoints(1);
		std::vector<Point3f> newObjPoints;

		CalculateCornerPositions(BOARDSIZE, SQUARESIZE, objectPoints[0]);

		// Do some math-y math for use case where our piece of paper is an imperfect planar target
		objectPoints[0][BOARDSIZE.width - 1].x = objectPoints[0][0].x + GRID_WIDTH;
		newObjPoints = objectPoints[0];
		objectPoints.resize(pointMatrix.size(), objectPoints[0]);

		// Ready some parameters
		int iFixedPoint = BOARDSIZE.width -1;
		cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
		cv::Mat distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
		cv::Mat rvecs, tvecs;

		// Calibration time!
		double rms;
		rms = cv::calibrateCameraRO(objectPoints, pointMatrix, imageSize, iFixedPoint, cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints, CALIBFLAG);

		// Tell us the overall RMS error
		std::cout << std::format("Calibration overall RMS re-projection error:\t{}\n", rms);

		// Check if everything went correctly
		bool ok = cv::checkRange(cameraMatrix) && cv::checkRange(distCoeffs);

		std::cout << "Camera Matrix:\t";
		std::cout << cameraMatrix;
		std::cout << "\n";

		std::cout << "Distortion Coefficients:\t";
		std::cout << distCoeffs;
		std::cout << "\n";
	}

	// Calculate the object points of the chessboard
	void CalculateCornerPositions(cv::Size boardSize, int squareSize, std::vector<Point3f>& out) 
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
	void DisplayCamera(cv::VideoCapture& webcam) {
		while (true) {
			cv::Mat stream; 
			webcam >> stream;
			cv::imshow("Stream", stream);
			cv::waitKey(1);
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
	cv::Mat img;
	ImageReader images;
	std::atomic<bool> imageReady = false;
	std::mutex img_mutex;
	clock_t prevTimestamp = 0;
	cv::Size imageSize;
	std::thread t1;
};


/* Note: The squares on the paper seem to be 22mm wide and long */
int main()
{
	// Using images
	Calibrator calibrator = Calibrator{IMAGES_PATH};

	// Using a webcam
	//Calibrator calibrator = Calibrator(0);

	calibrator.GatherData();
	std::cout << "Data gathered succesfully. Maybe. Hopefully.\n";

	calibrator.CalibrateCamera();

	//Click to exit
	cv::waitKey(0);
}
