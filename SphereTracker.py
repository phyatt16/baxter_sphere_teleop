import numpy as np
import cv2
import time
from scipy import stats
import rospy

from geometry_msgs.msg import Pose

np.set_printoptions(precision=2)

#Blue ball Hue: 100
#Orange ball Hue: 5
#Pink ball Hue: 170

class SphereTracker():

    def __init__(self,cap):

        rospy.init_node("sphere_tracker")

        self.left_sphere_pub = rospy.Publisher('teleop_master/left_pose',Pose,queue_size=10)
        self.right_sphere_pub = rospy.Publisher('teleop_master/right_pose',Pose,queue_size=10)        
        self.cap = cap
        self.left_z = 0
        self.right_z = 0
        
        ret, frame = self.cap.read()
        if(ret==True):
            self.last_frame = frame
            self.image_height = np.shape(frame)[0]
            self.image_width = np.shape(frame)[1]

        delta_t = 1/30.0

        # Left hand kalman filter
        # Constant Velocity Model
        self.left_kalman = cv2.KalmanFilter(6,3)
        self.left_kalman.measurementMatrix = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]],np.float32)
        self.left_kalman.transitionMatrix = np.array([[1,0,0,delta_t,0,0],[0,1,0,0,delta_t,0],[0,0,1,0,0,delta_t],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]],np.float32)

        self.left_kalman.processNoiseCov = np.array([[1.0/3.0*delta_t**3.0,0,0,1.0/2.0*delta_t**2.0,0,0],[0,1.0/3.0*delta_t**3.0,0,0,1.0/2.0*delta_t**2.0,0],[0,0,1.0/3.0*delta_t**3.0,0,0,1.0/2.0*delta_t**2.0],[1.0/2.0*delta_t**2.0,0,0,delta_t,0,0],[0,1.0/2.0*delta_t**2.0,0,0,delta_t,0],[0,0,1.0/2.0*delta_t**2.0,0,0,delta_t]],np.float32) * 100.0

        # Right hand kalman filter
        # Constant Velocity Model
        self.right_kalman = cv2.KalmanFilter(6,3)
        self.right_kalman.measurementMatrix = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]],np.float32)
        self.right_kalman.transitionMatrix = np.array([[1,0,0,delta_t,0,0],[0,1,0,0,delta_t,0],[0,0,1,0,0,delta_t],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]],np.float32)

        self.right_kalman.processNoiseCov = np.array([[1.0/3.0*delta_t**3.0,0,0,1.0/2.0*delta_t**2.0,0,0],[0,1.0/3.0*delta_t**3.0,0,0,1.0/2.0*delta_t**2.0,0],[0,0,1.0/3.0*delta_t**3.0,0,0,1.0/2.0*delta_t**2.0],[1.0/2.0*delta_t**2.0,0,0,delta_t,0,0],[0,1.0/2.0*delta_t**2.0,0,0,delta_t,0],[0,0,1.0/2.0*delta_t**2.0,0,0,delta_t]],np.float32) * 100.0

        # Setup termination criteria for cam/mean shift, either 10 iteration or move by atleast 1 pt
        self.term_crit = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 0, 10 )
        
        # Let the camera get its white balance adjusted
        start = time.time()
        while time.time()-start<1:
            # Capture frame by frame
            ret, frame = self.cap.read()


    def initialize(self):
        ret, frame = self.cap.read()
        
        # Select a region of interest for left hand
        c = 0
        r = 0
        w = int(self.image_width/4)
        h = self.image_height
        
        self.left_track_window = [c,r,w,h]
        self.left_tracked_point = np.array([c+w/2,r+h/2])

        # Set up the Kalman filter
        self.left_kalman.statePost = np.array([[self.left_tracked_point[0]],
                                               [self.left_tracked_point[1]],
                                               [0.0], # Assume at 0.0
                                               [0.0],
                                               [0.0],
                                               [0.0]],dtype=np.float32)
        self.left_kalman.errorCovPost = 1000.0*np.eye(6,dtype=np.float32)
        
        # Find the area0 for the left hand
        left_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Pick out the pink ball
        minHSV = np.array([160, 100, 100])
        maxHSV = np.array([180, 255, 255])
        maskHSV = cv2.inRange(left_hsv, minHSV, maxHSV)

        # kernel = np.ones((3,3),np.uint8)
        # maskHSV = cv2.erode(maskHSV,kernel,iterations = 2)
        # maskHSV = cv2.dilate(maskHSV,kernel,iterations = 2)

        # Get a measurement from self.left_kalman
        self.left_track_window = tuple(self.left_track_window)
        for i in xrange(0,500):
            ret, self.left_track_window = cv2.CamShift(maskHSV, self.left_track_window, self.term_crit)
        self.left_track_window = list(self.left_track_window)
        self.left_tracked_point = np.array([[self.left_track_window[0] + self.left_track_window[2]/2],
                                            [self.left_track_window[1] + self.left_track_window[3]/2],
                                            [0]],dtype=np.float32)

        
        left_area0 = ret[1][0]*ret[1][1]

        # Draw left area on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(frame,[pts],True,(0,255,0),2)        
        

        # Select a region of interest for right hand
        c = int(self.image_width*3.0/4.0)
        r = 0
        w = self.image_width
        h = self.image_height

        self.right_track_window = [c,r,w,h]
        self.right_tracked_point = np.array([c+w/2,r+h/2])

        # Set up the Kalman filter
        self.right_kalman.statePost = np.array([[self.right_tracked_point[0]],
                                                [self.right_tracked_point[1]],
                                                [0.0],
                                                [0.0],
                                                [0.0],
                                                [0.0]],dtype=np.float32)
        self.right_kalman.errorCovPost = 1000.0*np.eye(6,dtype=np.float32)
        
        # Find the area0 for the right hand
        right_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Pick out the blue ball
        minHSV = np.array([95, 100, 100])
        maxHSV = np.array([105, 255, 255])
        maskHSV = cv2.inRange(right_hsv, minHSV, maxHSV)
        
        # kernel = np.ones((3,3),np.uint8)
        # maskHSV = cv2.erode(maskHSV,kernel,iterations = 2)
        # maskHSV = cv2.dilate(maskHSV,kernel,iterations = 2)

        self.right_track_window = tuple(self.right_track_window)
        for i in xrange(0,500):
            ret, self.right_track_window = cv2.CamShift(maskHSV, self.right_track_window, self.term_crit)
        self.right_track_window = list(self.right_track_window)
        self.right_tracked_point = np.array([[self.right_track_window[0] + self.right_track_window[2]/2],
                                             [self.right_track_window[1] + self.right_track_window[3]/2],
                                             [0]],dtype=np.float32)
        
        right_area0 = ret[1][0]*ret[1][1]

        # Artificially erode these areas so the robot reaches in front
        self.right_area0 = 0.9*right_area0
        self.left_area0 = 0.9*left_area0

        # Draw right area on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(frame,[pts],True,(255,0,0),2)


        # -------------- Find the wingspan in pixels --------------------- #
        self.wingspan_pixels = float(abs(self.right_tracked_point[0] - self.left_tracked_point[0]))
        self.xy_gain = 2.0/self.wingspan_pixels
        self.height_pixels = 0.5*(self.right_tracked_point[1] + self.left_tracked_point[1])
        self.midpoint = [self.left_tracked_point[0] + self.wingspan_pixels/2.0, self.height_pixels]

        print self.left_tracked_point        
        print self.right_tracked_point
        print self.wingspan_pixels
        print self.height_pixels
        print self.midpoint
        
        # Draw the midpoint on the image
        frame[int(self.midpoint[1]-5):int(self.midpoint[1]+5),int(self.midpoint[0])-5:int(self.midpoint[0]+5),:] = 255

        cv2.imshow('Original Areas',frame)
        # cv2.waitKey(0)

    def track(self):
        # Capture frame by frame
        ret, frame = cap.read()
        
        if(ret==True):
            
            # For the left hand
            left_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Pick out the pink ball
            minHSV = np.array([160, 100, 100])
            maxHSV = np.array([180, 255, 255])
            maskHSV = cv2.inRange(left_hsv, minHSV, maxHSV)
            
            # kernel = np.ones((3,3),np.uint8)
            # maskHSV = cv2.erode(maskHSV,kernel,iterations = 2)
            # maskHSV = cv2.dilate(maskHSV,kernel,iterations = 2)
            
            #cv2.imshow("pink ball",maskHSV)                        
            
            # Predict the left hand state
            self.left_state = self.left_kalman.predict()
            self.left_track_window[0] = self.left_state[0] - self.left_track_window[2]/2
            self.left_track_window[1] = self.left_state[1] - self.left_track_window[3]/2
            self.left_track_window[2] = self.left_track_window[2]
            self.left_track_window[3] = self.left_track_window[3]
            self.left_z = self.left_state[2]
            
            # Get a measurement from self.left_kalman
            self.left_track_window = tuple(self.left_track_window)
            ret, self.left_track_window = cv2.CamShift(maskHSV, self.left_track_window, self.term_crit)
            
            self.left_track_window = list(self.left_track_window)
            
            area = ret[1][0]*ret[1][1]
            
            try:
                left_z_meas = 0.9*5000.0*(1.0/self.left_area0 - 1.0/area)
                self.left_tracked_point = np.array([[self.left_track_window[0] + self.left_track_window[2]/2],
                                                    [self.left_track_window[1] + self.left_track_window[3]/2],
                                                    [left_z_meas]],dtype=np.float32)
                
            except:
                print "Lost the left hand"


            # Correct the state based on a measurement
            self.left_state = self.left_kalman.correct(self.left_tracked_point)
            self.left_track_window[0] = self.left_state[0] - self.left_track_window[2]/2
            self.left_track_window[1] = self.left_state[1] - self.left_track_window[3]/2
            self.left_track_window[2] = self.left_track_window[2]
            self.left_track_window[3] = self.left_track_window[3]

                
                
            # Draw left on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            cv2.polylines(frame,[pts],True,(0,255,0),2)
            
            # For the right hand
            right_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Pick out the blue ball
            minHSV = np.array([95, 100, 100])
            maxHSV = np.array([105, 255, 255])
            maskHSV = cv2.inRange(right_hsv, minHSV, maxHSV)

            #cv2.imshow("blue ball",maskHSV)                        
            
            # Predict the right hand state
            self.right_state = self.right_kalman.predict()
            self.right_track_window[0] = self.right_state[0] - self.right_track_window[2]/2
            self.right_track_window[1] = self.right_state[1] - self.right_track_window[3]/2
            self.right_track_window[2] = self.right_track_window[2]
            self.right_track_window[3] = self.right_track_window[3]
            self.right_z = self.right_state[2]
        
            # Get a measurement from self.right_kalman
            self.right_track_window = tuple(self.right_track_window)
            ret, self.right_track_window = cv2.CamShift(maskHSV, self.right_track_window, self.term_crit)
            self.right_track_window = list(self.right_track_window)
            
            
            area = ret[1][0]*ret[1][1]
            
            try:
                right_z_meas = 0.9*5000.0*(1.0/self.right_area0 - 1.0/area)
                self.right_tracked_point = np.array([[self.right_track_window[0] + self.right_track_window[2]/2],
                                                     [self.right_track_window[1] + self.right_track_window[3]/2],
                                                     [right_z_meas]],dtype=np.float32)

            except:
                print "Lost the right hand"



            # Correct the state based on a measurement
            self.right_state = self.right_kalman.correct(self.right_tracked_point)
            self.right_track_window[0] = self.right_state[0] - self.right_track_window[2]/2
            self.right_track_window[1] = self.right_state[1] - self.right_track_window[3]/2
            # self.right_track_window[2] = self.right_state[0] + self.right_track_window[2]/2
            # self.right_track_window[3] = self.right_state[1] + self.right_track_window[3]/2
            self.right_track_window[2] = self.right_track_window[2]
            self.right_track_window[3] = self.right_track_window[3]

                
            # Draw right on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            cv2.polylines(frame,[pts],True,(255,0,0),2)
            
            cv2.imshow('Hands',frame)
                
            # ------ Solve for hands X,Y,Z -------- #
            self.left_x = self.xy_gain*(self.left_tracked_point[0]-self.midpoint[0])[0]
            self.left_y = -self.xy_gain*(self.left_tracked_point[1]-self.midpoint[1])[0]
            
            self.right_x = self.xy_gain*(self.right_tracked_point[0]-self.midpoint[0])[0]
            self.right_y = -self.xy_gain*(self.right_tracked_point[1]-self.midpoint[1])[0]
            
            
            print np.array([self.right_z[0], self.right_x, self.right_y]),"\t",np.array([self.left_z[0], self.left_x, self.left_y])
            
            # ----------- Publish the hand poses to ROS in Baxter's frame -------------- #
            right_pose = Pose()
            right_pose.position.x = self.right_z
            right_pose.position.y = self.right_x
            right_pose.position.z = self.right_y + .3
            
            right_pose.orientation.x = 0
            right_pose.orientation.y = 0
            right_pose.orientation.z = 0
            right_pose.orientation.w = 1
            
            left_pose = Pose()
            left_pose.position.x = self.left_z
            left_pose.position.y = self.left_x
            left_pose.position.z = self.left_y + .3
            
            left_pose.orientation.x = 0
            left_pose.orientation.y = 0
            left_pose.orientation.z = 0
            left_pose.orientation.w = 1
            
            
            print "Publishing"
            
            self.right_sphere_pub.publish(left_pose)
            self.left_sphere_pub.publish(right_pose)                
            
            print "Published"



if __name__ == '__main__':
        
    cap = cv2.VideoCapture(0)

    time.sleep(3)

    st = SphereTracker(cap)

    time.sleep(3)
    
    st.initialize()

    while not rospy.is_shutdown():
    
        st.track()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
