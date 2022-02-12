#include <format>
#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/filesystem.hpp>

using namespace cv;


class ImageReader
{
public:
	// Just use the filesystem iterator to find the images
	std::vector<cv::String> imageList;

	/* Constructor needs directory path
	*/
	ImageReader(std::string path)
	{
		// Read the directory and globs the found jpg's in a list
		cv::utils::fs::glob(path, "*.jpg", imageList, false);
	}
};

class Calibrator
{
public:
	std::vector<std::vector<Point2f>> pointMatrix;
	ImageReader images;

	Calibrator(std::string path) : images(path)
	{
	}

	/* Iterates every image in the given directory and gathers the chess corner points */
	void GatherData()
	{
		namedWindow("First CV Assignment", WINDOW_AUTOSIZE);
		cv::moveWindow("First CV Assignment", 0, 45);
		for (const cv::String image_path : images.imageList)
		{
			cv::Mat img = cv::imread(image_path);

			// The photos I took with my phone are 4000 x 3000, which makes finding the corners reeaallly slow, so we resize it
			cv::resize(img, img, Size(1600, 900));

			// Grayscale each image, color adds nothing
			cv::Mat gray;
			cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

			// Set some of the flags, finetuning can be done here
			const int chessFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;

			// Find the corners
			std::vector<Point2f> corners;
			bool ret = cv::findChessboardCorners(gray, cv::Size{ 6, 9 }, corners, chessFlags);
			if (ret)
			{
				std::cout << std::format("Chessboard corners found on image: {}\n", image_path);

				// Improve accuracy on corners with some subpixel magic
				cv::cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
				pointMatrix.push_back(corners);

				// Draw and show the corners for funsies
				cv::drawChessboardCorners(img, Size{ 6, 9 }, corners, ret);
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
};


/* Note: The squares on the paper seem to be 22mm wide and long */
int main()
{
	Calibrator calibrator = Calibrator{ "K:/Informatica/ComputerVision/Assignment01/imgs/" };
	calibrator.GatherData();
}
