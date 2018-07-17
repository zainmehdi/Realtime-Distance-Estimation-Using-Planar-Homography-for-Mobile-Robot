//
// Created by Zain on 18. 7. 17.
//

//
// Created by kari on 17. 12. 26.
//

// Program that uses IPM and world to screen calibration using checkerboard

#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv/cv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

/// Points
Point2f point;                            // point for mouse callback

/// Vectors
vector<Point2f> calibration_points;      // for storing calibration corners
vector<Point> line_points;               // vectors for storing path points
vector<Point2f> points[3];               // points for good features and optical flow
vector<Point> transformed_points;        // for storing  points of drawn path after transformation.

/// Flags
bool drag;
bool addRemovePt;
bool height_adjustment=false;
bool W2S_calibration=false;
bool first_run=true;
bool needToInit = false;
bool transformation_calculated=false;

/// Doubles and ints
double calibration_distance_x=22.8;  // in cm for angle of 64 degrees
double calibration_distance_y=17.3;  // in cm for angle of 64 degrees
double CF_x;                         // Multiplication factor for x axis distance
double CF_y;                         // Multiplication factor for y axis distance
double CF_x_default =0.31;
double CF_y_default =0.31;
const int MAX_COUNT = 100;            // Max numbers of features to find in image

/// Image Containers Mat
cv::Mat intrinsic, distortion;
cv::Mat gray_image,prev_gray, image1, image0,undistorted_image ;
cv::Mat birds_image,birds_image_c;
Mat gray, prevGray, image, frame;
Mat transformation;

void help(char *argv[]) {
    cout	<< "\nThis is my Master's thesis that's busting my ass right now ^_^ "
            << "\nIt uses IPM to warp the image to top view and uses Chess board to calibrate"
            << "\nbetween screen and world units"
            << "\nAfter Calibration we can get real world distance of object's in image from our robot"
            << "\n\nDevelopment still in progress"
            << endl;
}


void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ )
{
    if( event == EVENT_LBUTTONDOWN )
    {
        point = Point2f((float)x, (float)y);
        addRemovePt = true;
        drag=true;
    }

    else if( event == EVENT_MOUSEMOVE && drag)
    {


        line_points.push_back(CvPoint(x,y));
    }

    else if(event ==EVENT_LBUTTONUP)
    {
        drag=false;
    }
}
// args: [board_w] [board_h] [intrinsics.xml] [checker_image]
//
int main(int argc, char *argv[]) {



    /// Basic Parameters
    //
    // Chessboard Parameters:
    //
    int board_w = 8;
    int board_h = 6;
    int board_n = board_w * board_h;
    cv::Size board_sz(board_w, board_h);


    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size subPixWinSize(5,5), winSize(15,15);

    int i=1;


    // Intrinsic and Distortion Parameters
    //
    intrinsic.create(3,3,cv::DataType<double>::type);
    intrinsic.at<double>(0,0)=8.4028771356606751e+02;
    intrinsic.at<double>(0,1)=0;
    intrinsic.at<double>(0,2)=320;
    intrinsic.at<double>(1,0)=0;
    intrinsic.at<double>(1,1)=8.4282050609881685e+02;
    intrinsic.at<double>(1,2)=240;
    intrinsic.at<double>(2,0)=0;
    intrinsic.at<double>(2,1)=0;
    intrinsic.at<double>(2,2)=1;

    distortion.create(1,5,cv::DataType<double>::type);
    distortion.at<double>(0,0)=0.1185028650130305;
    distortion.at<double>(0,0)=-0.5150168858561729;
    distortion.at<double>(0,0)=0.0127408188974029;
    distortion.at<double>(0,0)=0.006831715564244691;
    distortion.at<double>(0,0)=0;

    /// Video Acquisition and Containers
    //

    VideoCapture cap(0);


    namedWindow("Birds_Eye", 1);                         // Window for displaying transformed image
    setMouseCallback( "Birds_Eye", onMouse, 0 );         // Mouse Callback
    // Image containers
    //
    cap >> frame;
    if( frame.empty() )
        return -1;

    frame.copyTo(image);
    cvtColor(image, gray, COLOR_BGR2GRAY);



    /// One time Initial Calibration
    //
    // UNDISTORT OUR IMAGE
    //
    cv::undistort(gray,undistorted_image, intrinsic, distortion, intrinsic);
//    cv::cvtColor(undistorted_image, gray_image, cv::COLOR_BGRA2GRAY);

    // GET THE CHECKERBOARD ON THE PLANE
    //
    vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners( // True if found
            image,                              // Input image
            board_sz,                           // Pattern size
            corners,                            // Results
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
    if (!found) {
        cout << "Couldn't acquire checkerboard on " << argv[1] << ", only found "
             << corners.size() << " of " << board_n << " corners\n";
        return -1;
    }

    // Get Subpixel accuracy on those corners
    //
    cv::cornerSubPix(
            gray,             // Input image
            corners,          // Initial guesses, also output
            cv::Size(11, 11), // Search window size
            cv::Size(-1, -1), // Zero zone (in this case, don't use)
            cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30,
                             0.1));


    /// This parts only needs to be done once. Its calibration between real and pixel world
    // GET THE IMAGE AND OBJECT POINTS:
    // Object points are at (r,c):
    // (0,0), (board_w-1,0), (0,board_h-1), (board_w-1,board_h-1)
    // That means corners are at: corners[r*board_w + c]
    /*
     *      (0,0)             (board_w-1,0)
     *       *---------------*
     *       |               |
     *       |               |
     *       |               |
     *       |               |
     *       |               |
     *       |               |
     *       *---------------*
     *      (0,board_h-1)    (board_w-1,board_h-1)
     */


    cv::Point2f objPts[4], imgPts[4];
    objPts[0].x = 0;
    objPts[0].y = 0;
    objPts[1].x = board_w - 1;
    objPts[1].y = 0;
    objPts[2].x = 0;
    objPts[2].y = board_h - 1;
    objPts[3].x = board_w - 1;
    objPts[3].y = board_h - 1;
    imgPts[0] = corners[0];
    imgPts[1] = corners[board_w - 1];
    imgPts[2] = corners[(board_h - 1) * board_w];
    imgPts[3] = corners[(board_h - 1) * board_w + board_w - 1];

    // DRAW THE POINTS in order: B,G,R,YELLOW
    //
    cv::circle(image, imgPts[0], 9, cv::Scalar(255, 0, 0), 3);
    cv::circle(image, imgPts[1], 9, cv::Scalar(0, 255, 0), 3);
    cv::circle(image, imgPts[2], 9, cv::Scalar(0, 0, 255), 3);
    cv::circle(image, imgPts[3], 9, cv::Scalar(0, 255, 255), 3);
    cv::imshow("Corners", image);



    // DRAW THE FOUND CHECKERBOARD
    //
    cv::drawChessboardCorners(gray, board_sz, corners, found);
    cv::imshow("Successful Detection", gray);

    cout<<"/nPress any key to continue\n";
    waitKey();

    // FIND THE HOMOGRAPHY
    // Once H is calculated we can use it without needing to calculate it again and again as the plane remains the same
    cv::Mat H = cv::getPerspectiveTransform(objPts, imgPts);

    // LET THE USER ADJUST THE Z HEIGHT OF THE VIEW
    //
    cout << "\nPress 'd' for lower birdseye view, and 'u' for higher (it adjusts the apparent 'Z' height), Esc to exit" << endl;
    double Z = 15;






    /// Main loop that does all

    for (;;) {



        ///////////////////////////////////////////////////////////////////////
        ///////////////////// Calibration Stuff//////////////////////////////

        /// World to screen calibration

        while (!W2S_calibration)
        {


            while(!height_adjustment)
            {
                // escape key stops
                H.at<double>(2, 2) = Z;
                // USE HOMOGRAPHY TO REMAP THE VIEW
                //

                cv::warpPerspective(gray,			// Source image
                                    birds_image, 	// Output image
                                    H,              // Transformation matrix
                                    gray.size(),   // Size for output image
                                    cv::WARP_INVERSE_MAP | cv::INTER_LINEAR,
                                    cv::BORDER_CONSTANT, cv::Scalar::all(0) // Fill border with black
                );

                cv::imshow("Birds_Eye", birds_image);
                int key = cv::waitKey(1);
                if (key == 'u')
                    Z += 0.5;
                if (key == 'd')
                    Z -= 0.5;
                if (key == 'c')
                    line_points.clear();
                if (key == 27)
                {
                    cout<<"\n Height adjustment done\n";
                    cout<<"\nW2S Calibration routine has started kindly select 3 extreme corners of checkerboard and press Esc \n";
                    height_adjustment=true;
                    break;
                }
            }



            if(addRemovePt)
            {
                circle( birds_image,point, 2, Scalar(0,255,255), -1, 8);
                cv::imshow("Birds_Eye", birds_image);
                calibration_points.push_back(point);
                addRemovePt = false;
            }

            char c = (char)waitKey(10);
            if( c == 27 && calibration_points.size()==3 )
            {

                /* Calculating factor which can recover distance of point in image from camera
                 * Lets say we know point in the real world in meters and we know its corresponding
                 * pixel in image (in this case we mark those points using mouse). Then whenever
                 * we get any other point in the image and we want to calculate its corresponding
                 * world coordinate. We can just multiply that pixel coordinate with scale factor
                 * and it will give us world coordinate. We have already calculated homography
                 * from world to image plane and since we are moving on a plane the relation wont
                 * break and all the points can be transformed from pixel to world point using
                 * same scaling factor.
                 */
                CF_x = abs(calibration_distance_x/(calibration_points[0].x-calibration_points[1].x));
                CF_y = abs(calibration_distance_y/(calibration_points[0].y-calibration_points[2].y));

                cout<<"\nCalibration Done\n"
                    <<"X factor ="<<CF_x<<"\n"
                    <<"Y factor ="<<CF_y<<"\n";
                W2S_calibration=true;

                break;
            }

            if( c == 27)
            {

                CF_x=CF_x_default;
                CF_y=CF_y_default;

                cout<<"\nUsing Default Calibration Parameters\n"
                    <<"X factor ="<<CF_x<<"\n"
                    <<"Y factor ="<<CF_y<<"\n";
                W2S_calibration=true;

                break;
            }



        }



        namedWindow("Tracked", 1);                         // Window for displaying transformed image
        setMouseCallback( "Tracked", onMouse, 0 );         // Mouse Callback


        cap >> frame;
        if( frame.empty() )
            break;

        frame.copyTo(image);
        cvtColor(image, gray, COLOR_BGR2GRAY);


        //////////////////////////////////////////////////////////////////////
        ///////////////////// Finding Good Features //////////////////////////


        if (needToInit) {
            // automatic initialization
//            Mat mask=Mat::zeros(birds_image.size(),CV_8U);
//            Mat roi=Mat(mask,Rect(point.x-50/2,point.y-50/2,50,50));
//            roi= Scalar(255, 255, 255);
            goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
            cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);
            needToInit = false;
        }

            //////////////////////////////////////////////////////////////////////
            ///////////////////// Calculating Optical Flow //////////////////////////
        else if(!points[0].empty())
        {
            vector<uchar> status;
            vector<float> err;
            if(prev_gray.empty())
                gray.copyTo(prev_gray);

            calcOpticalFlowPyrLK(prev_gray, gray, points[0], points[1], status, err, winSize,
                                 3, termcrit, 0, 0.001);


            /// Find rigid transformation between points of optical flow and use it to transform your own points
            transformation=estimateRigidTransform(points[0],points[1],true);
            transformation_calculated=true;
            size_t i, k;
            for( i = k = 0; i < points[1].size(); i++ )
            {

                if( !status[i] )
                    continue;

                points[1][k++] = points[1][i];
                circle( image, points[1][i], 3, Scalar(0,0,0), -1, 8);

            }
            perspectiveTransform(points[1],points[2],H.inv());
            points[1].resize(k);
        }

//        cout<<"\nDistance between Robot and point: "<<
        //////////////////////////////////////////////////////////////////////
        ///////////////////// Drawing Path in Image //////////////////////////

        if(!line_points.empty()&& transformation_calculated)
        {
            transform(line_points,transformed_points,transformation);
            line_points=transformed_points;
        }


        for(auto index:line_points)
        {

            circle(image,Point(index), 5, CV_RGB(255, 255, 0),1,8,0);


        }



        ///////////////////////////////////////////////////////////////////////
        ///////////////////// Perspective Transformation///////////////////////

        // escape key stops
        H.at<double>(2, 2) = Z;
        // USE HOMOGRAPHY TO REMAP THE VIEW
        //
        circle( gray, Point_<double>(640/2,475), 15, Scalar(0,0,0), -1, 8);


        Mat Pers_gray,test;
        cvtColor(image, Pers_gray, COLOR_BGR2GRAY);

        cv::warpPerspective(Pers_gray,			// Source image
                            birds_image, 	// Output image
                            H,              // Transformation matrix
                            gray.size(),   // Size for output image
                            cv::WARP_INVERSE_MAP | cv::INTER_LINEAR,
                            cv::BORDER_CONSTANT, cv::Scalar::all(0) // Fill border with black
        );

        cv::warpPerspective(gray,			// Source image
                            test, 	// Output image
                            H,              // Transformation matrix
                            gray.size(),   // Size for output image
                            cv::WARP_INVERSE_MAP | cv::INTER_LINEAR,
                            cv::BORDER_CONSTANT, cv::Scalar::all(0) // Fill border with black
        );

        birds_image.copyTo(birds_image_c);


        if(!points[2].empty())
        {

            for(auto index:points[2])
            {

                circle(test,Point(index), 5, CV_RGB(255, 255, 0),1,8,0);


            }
        }




//        cv::cvtColor(birds_image,birds_image_c,cv::COLOR_BayerBG2RGB);
        cv::imshow("Birds_Eye",birds_image_c );
        cv::imshow("Tracked",image );
        cv::imshow("Test check",test );
        int key = cv::waitKey(1);
        if (key == 'u')
            Z += 0.5;
        if (key == 'd')
            Z -= 0.5;
        if (key == 'c')
        {
            line_points.clear();
            points[0].clear();
            points[1].clear();
            transformation_calculated=false;
        }

        if (key == 'r')
        {
            needToInit=true;
        }

        if (key == 27)
            break;


        std::swap(points[1], points[0]);
        cv::swap(prev_gray, gray);



        first_run=false;

    }


    ///////////////////////////////////////////////////////////////////////
    ///////////////////// Rotation and Translation////////////////////////


    // SHOW ROTATION AND TRANSLATION VECTORS
    // We don't need them right now but they can be useful for finding distance between camera
    // and object provided we know the points of interest in object's coordinate frame
    //
    vector<cv::Point2f> image_points;
    vector<cv::Point3f> object_points;
    for (int i = 0; i < 4; ++i) {
        image_points.push_back(imgPts[i]);
        object_points.push_back(cv::Point3f(objPts[i].x, objPts[i].y, 0));
    }
    cv::Mat rvec, tvec, rmat;
    cv::solvePnP(object_points, 	// 3-d points in object coordinate
                 image_points,  	// 2-d points in image coordinates
                 intrinsic,     	// Our camera matrix
                 cv::Mat(),     	// Since we corrected distortion in the
            // beginning,now we have zero distortion
            // coefficients
                 rvec, 			    // Output rotation *vector*.
                 tvec  			    // Output translation vector.
    );
    cv::Rodrigues(rvec, rmat);

    // PRINT AND EXIT
    cout << "rotation matrix: " << rmat << endl;
    cout << "translation vector: " << tvec << endl;
    cout << "homography matrix: " << H << endl;
    cout << "inverted homography matrix: " << H.inv() << endl;

    return 1;
}