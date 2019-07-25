#include<iostream>
#include<opencv2\opencv.hpp>

#define displayMatImage(a,image) namedWindow(a, CV_WINDOW_NORMAL); imshow(a, image);

using namespace std;
using namespace cv;

struct Node
{
	Mat label;
	Node *next;
};

class Queue
{
public:
	Queue();
	void push(Mat image);
	void pop();
	Mat returnFront();
	size_t size();

private:
	Node *front, *rear;
};

Queue::Queue()
{
	front  = rear = NULL;
}

void Queue::push(Mat image){
	Node *temp = new Node;
	temp->label = image;
	temp->next = NULL;

	if(front==NULL){
		front = rear = temp;
		return;
	}
	else
	{
		rear->next = temp;
		rear=temp;
		return;
	}
}

void Queue::pop(){
	Node *temp;
	temp = front;
	front = front->next;
	delete temp;
}

Mat Queue::returnFront(){
	return front->label;
}

size_t Queue::size(){
	Node *temp;
	temp=front;
	size_t i=0;
	while(temp!=NULL){
		temp=temp->next;
		i++;
	}
	return i;
}