/*
Segmentation code

Created By:- Tript
		On:- 28 Jan 17
*/

#include<iostream>
#include<math.h>
#include<opencv2\opencv.hpp>
#include<vector>
#include"Queue.h"
#include<conio.h>

using namespace std;
using namespace cv;

//for loop
#define For(a,b) for(size_t a=0; a<(size_t)(b); ++a)
//reading data from 3 channel image
#define uData_3d(image, i , j,c) image.ptr<uchar>(i)[3*j + c]
//reading data from 1 channel image
#define uData_1d(image, i , j) image.at<uchar>(i,j)
//Display Image
#define displayMatImage(a,image) namedWindow(a, CV_WINDOW_NORMAL); imshow(a, image);


// for euc distance in segmentation
#define diff1(i,j) i-j
#define sqr(a) pow(a,2.0)
#define eucDistance(x1,x2,y1,y2,z1,z2) sqrt(sqr(diff1(x1,x2)) + sqr(diff1(y1,y2))+ sqr(diff1(z1,z2)))

// for circularity of contour
#define circularity(a) sqrt((4*3.14*contourArea(a))/arcLength(a,false))


const unsigned int BORDER = 15;
const unsigned int BORDER2 = 30;


/*void BlobDetector(Mat Image){
	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = 150;
	params.maxThreshold = 255;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 500;
	params.maxArea = 5000;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.4;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.4;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.1;


	// Storage for blobs
	vector<KeyPoint> keypoints;

	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// Detect blobs
	detector->detect( Image, keypoints);

	// Draw detected blobs as green circles.
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
	// the size of the circle corresponds to the size of blob

	Mat im_with_keypoints;
	drawKeypoints( Image, keypoints, im_with_keypoints, Scalar(0,255,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

	// Show blobs
	displayMatImage("Blob Detector", im_with_keypoints);imwrite("blob detector.jpg",im_with_keypoints);

	/*for(int i=0; i<keypoints.size; i++)
	{
		Point2f a = keypoints[i].pt;

	}
}*/

void contours(Mat image, Mat original){
	// find the contours
    vector< vector<Point> > contours;

	findContours(image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // you could also reuse img1 here
	Mat mask = Mat::zeros(image.rows, image.cols, CV_8UC1);

    // CV_FILLED fills the connected components found
	drawContours(mask, contours, -1, Scalar(255), CV_FILLED);
	displayMatImage("mask filled",mask);

    // normalize so imwrite(...)/imshow(...) shows the mask correctly!
    normalize(mask.clone(), mask, 0.0, 255.0, CV_MINMAX, CV_8UC1);


	vector<vector<Point> > contours_poly( contours.size() );
	vector<Rect> boundRect( contours.size() );

	cout<<contours.size();
	For(i,contours.size())
     {
		 approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
		 boundRect[i] = boundingRect( Mat (contours_poly[i]) );
     }

	/*vector<vector<Point> >hull( contours.size() );
	for( size_t i = 0; i < contours.size(); i++ )
     {   convexHull( Mat(contours[i]), hull[i], false ); }*/
#ifndef top
	Queue roi;
	for(size_t i=0;i<contours.size();i++)
	{
		if(contourArea(contours[i])>250 && contourArea(contours[i])<20000){
			//if(isContourConvex(contours[i]))
					double circular=circularity(contours[i]);
					if(circular>0.5){
					Rect rect (boundRect[i].tl().x-BORDER,boundRect[i].tl().y-BORDER,boundRect[i].width+BORDER2, boundRect[i].height+BORDER2);

					bool is_inside = (rect & Rect(0, 0, original.cols, original.rows)) == rect;
					String size;
					if(is_inside){
						//rectangle(original, rect, (0,0,255), 2, 8, 0 );
						Mat crop = original(rect);
						roi.push(crop);
						size = to_string(roi.size());
						displayMatImage("roi "+size,crop); imwrite("roi "+size+".jpg",crop);
					}else{
						//rectangle(original, boundRect[i].tl(), boundRect[i].br(), (0,0,255), 2, 8, 0 );
						Rect rect (boundRect[i].tl().x,boundRect[i].tl().y,boundRect[i].width, boundRect[i].height);
						Mat crop = original(rect); imwrite("roi.jpg",crop);

						roi.push(crop);
						size = to_string(roi.size());
						displayMatImage("roi "+size,crop); imwrite("roi "+size+".jpg",crop);
					}
				}else continue;
			}else continue;
	}
	#endif
	/*int j = roi.size();
	for(int i=0;i<j;i++){
		Mat holder = roi.front();
		displayMatImage("roi "+i,holder);
		roi.pop();
	}*/
	//displayMatImage("Contours",mask);

}

void saliencyMap(Mat image){
	int image_rows = image.rows,
		image_cols = image.cols;

	cout<<image.rows<<"   "<<image.cols;
	vector <double> meanVal(3);
	meanVal[0]= 0;
	meanVal[1]= 0;
	meanVal[2]= 0;

	//resize(image, image, Size(image_rows*0.5,image_cols*0.5));

	int rows = image.rows,
		cols = image.cols,
		totalPixel = rows*cols,
		type = image.type();
	Size size = image.size();

	Mat temp,temp2,image_pyrMeanShift,image_smooth,canny,
		gradient = Mat(size,CV_8UC1),
		saliency = Mat(size,CV_8UC1);


	//Smoothening (Removing salt and pepper)
	medianBlur(image,image_smooth,21);

	cvtColor(image,temp,CV_RGB2Lab);

	pyrMeanShiftFiltering(image_smooth, image_pyrMeanShift,10,6,6);
	//displayMatImage("PyrMeanShift", image_pyrMeanShift); //imwrite("mean_shift.jpg",image_pyrMeanShift);

	image_pyrMeanShift  = image_pyrMeanShift*1.2;
	displayMatImage("PyrMeanShift2", image_pyrMeanShift); //imwrite("mean_shift.jpg",image_pyrMeanShift);

	For(i , rows){
		For(j, cols){

			meanVal[0] += uData_3d(image,i,j,0);
			meanVal[1] += uData_3d(image,i,j,1);
			meanVal[2] += uData_3d(image,i,j,2);
		}
	}

	Scalar mean;
	mean.val[0] = meanVal[0]/totalPixel;
	mean.val[1] = meanVal[1]/totalPixel;
	mean.val[2] = meanVal[2]/totalPixel;

	For(i,rows){
		For(j, cols){
			saliency.at<uchar>(i,j) = eucDistance(uData_3d(image_pyrMeanShift,i,j,0),mean.val[0],uData_3d(image_pyrMeanShift,i,j,1),mean.val[1],uData_3d(image_pyrMeanShift,i,j,2),mean.val[2]);
		}
	}
	normalize(saliency, saliency,0,255, NORM_MINMAX);
	//displayMatImage("saliency map", saliency); imwrite("saliency.jpg", saliency);

	Canny(saliency,saliency,100,255);
	//displayMatImage("Canny map", saliency); imwrite("canny.jpg",saliency);

	/*vector<Mat> B_G_R_normalized(3);
    split(image_pyrMeanShift, B_G_R_normalized);

    vector<Mat> bt(4);

    bt[3] = Mat(size, CV_8UC1); //Yellow Channel
    bt[2] = Mat(size, CV_8UC1); //Red channel
    bt[1] = Mat(size, CV_8UC1); //Green channel
    bt[0] = Mat(size, CV_8UC1); //Blue channel

    unsigned int b1, g1, r1;

        For(i, rows)
        {
            For(j, cols)
            {

                b1 = uData_1d(B_G_R_normalized[0], i, j);
                g1 = uData_1d(B_G_R_normalized[1], i, j);
                r1 = uData_1d(B_G_R_normalized[2], i, j);

                if((b1+r1+g1)==0u)
                {
                    uData_1d(B_G_R_normalized[0], i, j) = 255;
                    uData_1d(B_G_R_normalized[1], i, j) = 255;
                    uData_1d(B_G_R_normalized[2], i, j) = 255;
                }
                else
                {
                    uData_1d(B_G_R_normalized[0], i, j) = (b1*255u)/(b1+g1+r1);
                    uData_1d(B_G_R_normalized[1], i, j) = (g1*255u)/(b1+g1+r1);
                    uData_1d(B_G_R_normalized[2], i, j) = (r1*255u)/(b1+g1+r1);
                }
                uData_1d(bt[0],i,j) = abs( b1 - (r1+g1)/2u);
                uData_1d(bt[1],i,j) = abs( g1 - (r1+b1)/2u);
                uData_1d(bt[2],i,j) = abs( r1 - (b1+g1)/2u);
                if(r1>g1)
                    uData_1d(bt[3],i,j) = abs(((r1+g1) - (r1-g1))/2u);
                else
                    uData_1d(bt[3],i,j) = abs(((r1+g1) - (g1-r1))/2u);

            }
    }

		{
			Canny(B_G_R_normalized[0],B_G_R_normalized[0],50,100);
			Canny(B_G_R_normalized[1],B_G_R_normalized[1],50,100);
			Canny(B_G_R_normalized[2],B_G_R_normalized[2],50,100);
			Canny(bt[0],bt[0],50,100);
			Canny(bt[1],bt[1],50,100);
			Canny(bt[2],bt[2],50,100);
			Canny(bt[3],bt[3],50,100);
		}

		/*{
        displayMatImage("R norm",B_G_R_normalized[2]);imwrite("R caanny norm.jpg", B_G_R_normalized[0]);
        displayMatImage("G norm",B_G_R_normalized[1]);imwrite("G canny norm.jpg", B_G_R_normalized[1]);
        displayMatImage("B norm",B_G_R_normalized[0]);imwrite("B canny norm.jpg", B_G_R_normalized[2]);
        displayMatImage("bt blue",bt[0]);imwrite("R canny.jpg", bt[0]);
        displayMatImage("bt green",bt[1]);imwrite("G canny.jpg", bt[1]);
        displayMatImage("bt red",bt[2]);imwrite("B canny.jpg", bt[2]);
        displayMatImage("bt yellow",bt[3]);imwrite("Y canny.jpg", bt[3]);
    }

	Mat finalImage = Mat(size,CV_8UC1);
	finalImage = saliency+B_G_R_normalized[0]+B_G_R_normalized[1]+B_G_R_normalized[2]+bt[0]+bt[1]+bt[2]+bt[3];
	//displayMatImage(" Final Image ", finalImage);imwrite("final.jpg", finalImage);*/

	//Gradient
	dilate(saliency,temp,Mat());
	erode(saliency,temp2,Mat());
	For(i,rows){
		For(j,cols){
			gradient.at<uchar>(i,j) = temp.at<uchar>(i,j) - temp2.at<uchar>(i,j);
		}
	}

	//displayMatImage("Gradient Image", gradient); imwrite("gradient.jpg",gradient);
	gradient = gradient + saliency;
	displayMatImage("Gradient+Canny Img",gradient);

	//resize(image, image, Size(image_rows, image_cols));

	contours(gradient, image);
}


int main(){
	Mat img;
	img = imread("C:\\Users\\Tript\\Desktop\\dsad.jpg",1);
	if(img.cols==0){cout<<"Image not read"; getchar(); return 0;}
	saliencyMap(img);

	displayMatImage("orignal image", img);imwrite("orig.jpg",img);

	waitKey(0);
	return 0;
}
