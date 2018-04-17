#!/usr/bin/env python
import sys
import argparse
import baxter_interface
import rospy
import numpy as np
import socket
import time
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,)
from baxter_core_msgs.msg import DigitalIOState
import baxter_pykdl as kdl
from std_msgs.msg import Header
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
import tf
import baxter_left_kinematics as blk
import baxter_right_kinematics as brk

rospy.init_node("slave")

rate = rospy.Rate(1000)

left = baxter_interface.Limb('left')
right = baxter_interface.Limb('right')

ileft = kdl.baxter_kinematics('left')
iright = kdl.baxter_kinematics('right')
ik = kdl.baxter_kinematics.inverse_kinematics

left.set_command_timeout(5)
right.set_command_timeout(5)
left.set_joint_position_speed(.25)    #Set a value (0-1) to control speed. 1 is fast.
right.set_joint_position_speed(.25)    #Set a value (0-1) to control speed. 1 is fast.

left_gripper = baxter_interface.Gripper('left')
right_gripper = baxter_interface.Gripper('right')

left_gripper.calibrate(True, 5)
right_gripper.calibrate(True, 5)

def skew(d):
    return np.matrix([[0., -d[2], d[1]], [d[2], 0., -d[0]], [-d[1], d[0], 0.]])
    
def biggest_mag(lst):
    biggest_mag = 0
    for i in range(lst.__len__()):
        num = np.sqrt(np.square(lst[i]))
        if num > np.sqrt(np.square(biggest_mag)):
            biggest_mag = num
    return biggest_mag
    
def lst_diff(lst_a, lst_b):
    new_lst = []
    for i in range(lst_a.__len__()):
        new_lst.append(lst_a[i] - lst_b[i])
    return new_lst
    
def lst_sum(lst_a, lst_b):
    new_lst = []
    for i in range(lst_a.__len__()):
        new_lst.append(lst_a[i] + lst_b[i])
    return new_lst
    
def sum_lst(lst_a):
    num = 0
    for i in range(lst_a.__len__()):
        num = num + lst_a[i]
    return num
    
def quat_to_list(quat):
    lst = []
    lst.append(quat.x)
    lst.append(quat.y)
    lst.append(quat.z)
    lst.append(quat.w)
    return lst
    
def point_to_list(point):
    lst = []
    lst.append(point.x)
    lst.append(point.y)
    lst.append(point.z)
    return lst
    
def pos_diff(pose_a, pose_b):
    new_pose = Pose()
    new_pose.position.x = pose_b.position.x - pose_a.position.x
    new_pose.position.y = pose_b.position.y - pose_a.position.y
    new_pose.position.z = pose_b.position.z - pose_a.position.z
    return new_pose
    
def greatest_diff(lst_a, lst_b):
    big_diff = 0
    for i in range(lst_a.__len__()):
        diff = lst_a[i] - lst_b[i]
        if np.sqrt(np.square(diff)) > np.sqrt(np.square(big_diff)):
            big_diff = np.sqrt(np.square(diff))
    return big_diff
    
def joint_angles_to_pose(limb, joint_angles):
    if limb == left:
        bk = blk
    elif limb == right:
        bk = brk
    ee_matrix = bk.FK[6](joint_angles)
    new_pose = Pose()
    new_pose.position.x = ee_matrix[0][3]
    new_pose.position.y = ee_matrix[1][3]
    new_pose.position.z = ee_matrix[2][3]
    new_pose.orientation = tf.transformations.quaternion_from_matrix(ee_matrix)
    return new_pose
    
def quat_divide(pose_a, pose_b):
    #Divides pose_a by pose_b
    divided = tf.transformations.quaternion_multiply(quat_to_list(pose_a.orientation), tf.transformations.quaternion_inverse(pose_b.orientation))
    return divided
    
def delta_x_and_e(pose_b, pose_a):

    position_diff = pos_diff(pose_a, pose_b)
    quat_divided = quat_divide(pose_b, pose_a)
    
    delta_epsilon = []
    delta_epsilon.append(position_diff.position.x)
    delta_epsilon.append(position_diff.position.y)
    delta_epsilon.append(position_diff.position.z)
    delta_epsilon.append(quat_divided[0])
    delta_epsilon.append(quat_divided[1])
    delta_epsilon.append(quat_divided[2])
    return delta_epsilon
    
def analytical_jacobian(limb, joint_angles):
    if limb == left:
        bk = blk
    elif limb == right:
        bk = brk
    pose = joint_angles_to_pose(limb, joint_angles)
    Jac = bk.J[6](joint_angles)
    epsilon = []
    eita = pose.orientation[3]
    epsilon.append(pose.orientation[0])
    epsilon.append(pose.orientation[1])
    epsilon.append(pose.orientation[2])
    p = np.array(.5*((eita*np.eye(3,3))-skew(epsilon)))
    a = np.hstack((np.eye(3,3), np.zeros((3,3))))
    b = np.hstack((np.zeros((3,3)), p))
    J_t = np.vstack((a,b)).dot(Jac)
    return J_t
    
def solve_local_ik(limb, goal_pose):
    #Create a list of joint angles
    limb_joint_angles = []
    for i in limb.joint_names():
        limb_joint_angles.append(limb.joint_angle(i))

    count = 0
    diff = 10
    q_next = []
    goal = goal_pose
    q_curr = limb_joint_angles
    deltas = delta_x_and_e(goal, joint_angles_to_pose(limb, q_curr))
    if biggest_mag(deltas) <=.01:
        print "I'm not solving because my biggest error is: ", biggest_mag(deltas)
        t=0
        cmd_angles = {}
        #Convert joint list into a dictionary
        for i in limb.joint_names():
            cmd_angles[i] = q_curr[t]
            t=t+1
        return cmd_angles
    ik_start = time.time()

    while diff > .0001:
        a = np.hstack(((.75*np.eye(3,3)), np.zeros((3,3))))
        b = np.hstack((np.zeros((3,3)), (.75*np.eye(3,3))))
        constants = np.vstack((a,b))
        deltas = np.array(delta_x_and_e(goal, joint_angles_to_pose(limb, q_curr)))
        deltas = deltas.dot(constants)
        J_t = analytical_jacobian(limb, q_curr)
        q_next = lst_sum(q_curr, (.1)*np.dot(J_t.T, deltas))
        
        #Establish Joint Limits
        if limb == left:
            if q_next[0]<-0.89:
                q_next[0] = -.89
            if q_next[0] > 2.461:
                q_next[0] = 2.461
        elif limb == right:
            if q_next[0]>0.89:
                q_next[0] = .89
            if q_next[0] < -2.461:
                q_next[0] = -2.461
                
        if q_next[1] < -2.147:
            q_next[1] = -2.147
        elif q_next[1] > 1.047:
            q_next[1] = 1.047
            
        if q_next[2] < -3.028:
             q_next[2] = -3.028
        elif q_next[2] > 3.028:
            q_next[2] = 3.028
            
        if q_next[3] < -0.052:
            q_next[3] = -0.052
        elif q_next[3] > 2.618:
            q_next[3] = 2.618
            
        if q_next[4] < -3.059:
            q_next[4] = -3.059
        elif q_next[4] > 3.059:
            q_next[4] = 3.059
            
        if q_next[5] < -1.571:
            q_next[5] = -1.571
        elif q_next[5] > 2.094:
            q_next[5] = 2.094
            
        if q_next[6] < -3.059:
            q_next[6] = -3.059
        elif q_next[6] > 3.059:
            q_next[6] = 3.059

        deltas = delta_x_and_e(goal, joint_angles_to_pose(limb, q_next))
        diff = greatest_diff(q_next, q_curr)
        q_curr = q_next
        count = count + 1
        if count>1000:
            break
    ik_end = time.time()
    print "Total solve time: ", ik_end - ik_start
    print "Iterations: \n", count
    print limb.name + " Goal Pose: \n", goal
    print "Solution Pose: \n", joint_angles_to_pose(limb, q_next)
    cmd_angles = {}
    t=0
    for i in limb.joint_names():
        cmd_angles[i] = q_next[t]
        t=t+1
    return cmd_angles

def ik_solve(limb, pos, orient):   
    start_time = time.time()
    ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK, True)
    ikreq = SolvePositionIKRequest()
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    poses = {
        str(limb): PoseStamped(header=hdr,
            pose=Pose(position=pos, orientation=orient))}
            
    ikreq.pose_stamp.append(poses[limb])
    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException), e:
        rospy.logerr("Service call failed: %s" % (e,))
        return 1
    if (resp.isValid[0]):
        print("SUCCESS - Valid Joint Solution Found for the " +str(limb)+ " arm")
        # Format solution into Limb API-compatible dictionary
        limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        return limb_joints
    else:
        print("INVALID POSE - No Valid Joint Solution Found for the " +str(limb)+ " arm")
        return 1
    end_time = time.time()
    ik_time = end_time - start_time
    print "Time to solve: ", ik_time
    return -1
    
def left_ik_cb(pose):
    #array = ik(ileft,pose.position,pose.orientation)
    #if array != None:
    #    array = array[0]
    #    cmd_angles = {}
    #    t=0
    #    for i in left.joint_names():
    #        cmd_angles[i] = array[t]
    #        t=t+1
    #    print "Left Baxter pykdl angles: ", cmd_angles
    #    left.set_joint_positions(cmd_angles)
    left_ik_joint_angle = ik_solve('left', pose.position, pose.orientation)  
    if left_ik_joint_angle == 1:
        left_ik_joint_angle = solve_local_ik(left, pose) 
    left.set_joint_positions(left_ik_joint_angle)
    
def right_ik_cb(pose):
    #array = ik(iright,pose.position,pose.orientation)
    #if array != None:
    #    array = array[0]
    #    cmd_angles = {}
    #    t=0
    #    for i in right.joint_names():
    #        cmd_angles[i] = array[t]
    #        t=t+1
    #    print "Right Baxter pykdl angles: ", cmd_angles
    #    right.set_joint_positions(cmd_angles)
    right_ik_joint_angle = ik_solve('right', pose.position, pose.orientation)
    if right_ik_joint_angle == 1:
        right_ik_joint_angle = solve_local_ik(right, pose)
    right.set_joint_positions(right_ik_joint_angle)

def left_but_cb(state):
    if state.state == 1:
        print "Left Button Pressed"
        if left_gripper.position() > 50:
            left_gripper.close()
        if left_gripper.position() < 50:
            left_gripper.open()
    else:
        pass
        
def right_but_cb(state):
    if state.state == 1:
        print "Right Button Pressed"
        if right_gripper.position() > 50:
            right_gripper.close()
        if right_gripper.position() < 50:
            right_gripper.open()
    else:
        pass
    
def receiver():
    left_sub = rospy.Subscriber('Teleop_Master/Left_Pose', Pose, left_ik_cb, None, 1)
    right_sub = rospy.Subscriber('Teleop_Master/Right_Pose', Pose, right_ik_cb, None, 1)
    left_button_sub = rospy.Subscriber('Teleop_Master/Left_Button', DigitalIOState, left_but_cb, None, 1)
    right_button_sub = rospy.Subscriber('Teleop_Master/Right_Button', DigitalIOState, right_but_cb, None, 1)

    
    while not rospy.is_shutdown():
        rate.sleep()
            
def main():

    receiver()
    
if __name__=='__main__':
    sys.exit(main())
