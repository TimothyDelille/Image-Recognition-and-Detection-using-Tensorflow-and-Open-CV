#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal, Command
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
from pymavlink import mavutil

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
from PIL import Image
import display_map

import cv2
cap = cv2.VideoCapture('training_clip.mp4')

#-----------------------------
# ********* IMAGE RECOGNITION **********
#-----------------------------

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util

from utils import visualization_utils as vis_util

MODEL_NAME = 'DASSAULT_UAV_CHALLENGE'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')

NUM_CLASSES = 6

# Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# ## Helper code

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def get_target_distance(image, box, classe):
    """
    returns the distance between the center of the box and the center of the image (position of the drone) in m
    """
    im_width, im_height = image.shape[0], image.shape[1]
    image_center = np.array([im_width/2.,im_height/2.])
    
    ymin, xmin, ymax, xmax = box #ratio
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    box_center = np.array([(left+right)/2., (top+bottom)/2.])

    if classe=='red_cross' or classe=='yellow_cross' or classe=='blue_cross':
        box_length=((bottom-top)+(right-left))/2.
    elif classe=='red_line' or classe=='yellow_line' or classe=='blue_line':
        box_length=max(bottom-top,right-left)
    distance_px = box_center-image_center

    object_real_world_mm = 1000. #1000 mm for rectangle or cross
  
    dNorth = distance_px[1]*object_real_world_mm/(box_length*1000.)
    dEast = distance_px[0]*object_real_world_mm/(box_length*1000.)
    return [dNorth, dEast]

def draw_target_on_map(app,target,dNorth,dEast):
    dlocation = [dEast, dNorth]
    if target=='red_cross':
	app.draw_cross('red', dlocation, app.factor, app.drone_position)
    elif target=='yellow_cross':
	app.draw_cross('yellow', dlocation, app.factor, app.drone_position)
    elif target=='blue_cross':
	app.draw_cross('blue', dlocation, app.factor, app.drone_position)
    elif target=='red_line':
	app.draw_rectangle('red', dlocation, app.factor, app.drone_position)
    elif target=='yellow_line':
	app.draw_rectangle('yellow', dlocation, app.factor, app.drone_position)
    elif target=='blue_line':
    	app.draw_rectangle('blue', dlocation, app.factor, app.drone_position)    
#-----------------------------
# ********* MISSION **********
#-----------------------------

#Set up option parsing to get connection string
import argparse  
parser = argparse.ArgumentParser(description='Demonstrates basic mission operations.')
parser.add_argument('--connect', 
                   help="vehicle connection target string. If not specified, SITL automatically started and used.")
args = parser.parse_args()

connection_string = args.connect
sitl = None

#Start SITL if no connection string specified
if not connection_string:
    import dronekit_sitl
    sitl = dronekit_sitl.start_default()
    connection_string = sitl.connection_string()

# Connect to the Vehicle
print('Connecting to vehicle on: %s' % connection_string)
vehicle = connect(connection_string, wait_ready=True)
    
def get_location_metres(original_location, dNorth, dEast):
    earth_radius=6378137.0 #Radius of "spherical" earth
    #Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/earth_radius
    #dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))

    #New position in decimal degrees
    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    return LocationGlobalRelative(newlat, newlon,original_location.alt)


def get_distance_metres(aLocation1, aLocation2):
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5

def distance_to_current_waypoint():
    nextwaypoint = vehicle.commands.next
    if nextwaypoint==0:
        return None #home location
    missionitem=vehicle.commands[nextwaypoint-1] #commands are zero indexed
    lat = missionitem.x
    lon = missionitem.y
    alt = missionitem.z
    targetWaypointLocation = LocationGlobalRelative(lat,lon,alt)
    distancetopoint = get_distance_metres(vehicle.location.global_frame, targetWaypointLocation)
    return distancetopoint

def download_mission():
    cmds = vehicle.commands
    cmds.download()
    cmds.wait_ready() # wait until download is complete.

def adds_mission(landing, altitude):
    """
    The function assumes vehicle.commands matches the vehicle mission state 
    (you must have called download at least once in the session and after clearing the mission)
    """	

    cmds = vehicle.commands

    print(" Clear any existing commands")
    cmds.clear() 
    
    print(" Define/add new commands.")
    # Add new commands. The meaning/order of the parameters is documented in the Command class. 
     
    #Add MAV_CMD_NAV_TAKEOFF command. This is ignored if the vehicle is already in the air.
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 0, altitude))

    #Define the MAV_CMD_NAV_WAYPOINT location and add the commands
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0, 0, 0, 0, landing.lat, landing.lon, altitude))

    print(" Upload new commands to vehicle")
    cmds.upload()

def arm_and_takeoff(aTargetAltitude):
    print("Basic pre-arm checks")
    # Don't let the user try to arm until autopilot is ready
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:      
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude) # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command 
    #  after Vehicle.simple_takeoff will execute immediately).
    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)      
        if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95: #Trigger just below target alt.
            print("Reached target altitude")
            break
        time.sleep(1)

def goto(dNorth, dEast, gotoFunction=vehicle.simple_goto):
    """
    Moves the vehicle to a position dNorth metres North and dEast metres East of the current position.

    The method takes a function pointer argument with a single `dronekit.lib.LocationGlobal` parameter for 
    the target position. This allows it to be called with different position-setting commands. 
    By default it uses the standard method: dronekit.lib.Vehicle.simple_goto().

    The method reports the distance to target every two seconds.
    """
    
    currentLocation = vehicle.location.global_relative_frame
    targetLocation = get_location_metres(currentLocation, dNorth, dEast)
    targetDistance = get_distance_metres(currentLocation, targetLocation)
    gotoFunction(targetLocation)
    
    #print "DEBUG: targetLocation: %s" % targetLocation
    #print "DEBUG: targetLocation: %s" % targetDistance

    while vehicle.mode.name=="GUIDED": #Stop action if we are no longer in guided mode.
        #print "DEBUG: mode: %s" % vehicle.mode.name
        remainingDistance=get_distance_metres(vehicle.location.global_relative_frame, targetLocation)
        print("Distance to target: ", remainingDistance)
        if remainingDistance<=targetDistance*0.01: #Just below target, in case of undershoot.
            print("Reached target")
            break;
        time.sleep(2)
        
################################
##### START OF THE MISSION #####
################################

target_altitude = 10.
currentLocation = vehicle.location.global_relative_frame
landing_GPS = get_location_metres(currentLocation, 50, 10)

# STARTING tkinter GUI
drone_location = {'lat':currentLocation.lat, 'lon':currentLocation.lon}
waypoints = [drone_location, {'lat':landing_GPS.lat, 'lon':landing_GPS.lon}]
app = display_map.Application('single', drone_location, waypoints, 0)
        
print('Create a new mission (for current location)')
adds_mission(landing_GPS,target_altitude)

arm_and_takeoff(target_altitude)

print("Starting mission")
# Reset mission set to first (0) waypoint
vehicle.commands.next=0

# Set mode to AUTO to start mission
vehicle.mode = VehicleMode("AUTO")

vehicle.commands.next=1

# Detection
target = 'yellow_cross'
target_counter = pd.DataFrame(columns=['yellow_cross','red_cross','blue_cross','yellow_line','red_line','blue_line'])
seuil = 0.07

delivered = False
loop_counter = 0
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
    	    #update tkinter gui
            new_location = {'lat': vehicle.location.global_relative_frame.lat, 
			    'lon': vehicle.location.global_relative_frame.lon}
	    #print('vehicle lat:',new_location['lat'], ' and vehicle lon:', new_location['lon'])
	    app.move_drone(drone_location, new_location, app.factor, app.drone, vehicle.location.global_relative_frame.alt)
	    
	    drone_location = new_location
		
	    #detect targets
            ret, image_np = cap.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
		min_score_thresh=.45,
                line_thickness=8)

            cv2.imshow('object detection', cv2.resize(image_np, (400,400)))
        
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)
            min_score_thresh =.45
        
            candidates=[]
            candidates_scores=[]
            for i in range(boxes.shape[0]):
                if scores[i]>min_score_thresh:
                    box = tuple(boxes[i].tolist())
                    if classes[i] in category_index.keys():
                        if category_index[classes[i]]['name']==target:
                            candidates.append(i)
                            candidates_scores.append(scores[i])
			    #print('FOUND TARGET: ', target, ' with a ', int(scores[i]*100), '% probability')
			#else:
			    #print('FOUND WRONG TARGET: ', category_index[classes[i]]['name'], ' with a ', int(scores[i]*100), '% probability')
        
            if not candidates:
		target_counter = target_counter.append(pd.Series([0,0,0,0,0,0],index=target_counter.columns), ignore_index=True)
            else:
                candidates_scores = np.array(candidates_scores)
                target_box = tuple(boxes[candidates[np.argmax(candidates_scores)]].tolist())#choix du candidat avec le plus haut score

		currentLocation = vehicle.location.global_relative_frame
                dNorth_target, dEast_target = get_target_distance(image_np, target_box, target)
		target_location = get_location_metres(currentLocation, dNorth_target, dEast_target)
		
		target_counter = target_counter.append(pd.Series([1,0,0,0,0,0],index=target_counter.columns), ignore_index=True)
		loop_counter+=1
		
		moving_avg = target_counter[target].rolling(20).mean().fillna(0)
		if moving_avg.iloc[-1]>seuil and delivered == False:
		    	
			# Draw target on map		
			draw_target_on_map(app, target, dNorth_target, dEast_target)
			# Get the set of commands from the vehicle

			delivered = True

			cmds = vehicle.commands
			cmds.download()
			cmds.wait_ready()
			
			cmds.clear()

			cmds.add(Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0, 0, 0, 0, target_location.lat, target_location.lon, target_location.alt))
			cmds.add(Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 0, 0))
			# Insert "Deliver the package" command
			cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 0, target_altitude))
			cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0, 0, 0, 0, landing_GPS.lat, landing_GPS.lon, landing_GPS.alt))

			cmds.upload()
			vehicle.commands.next=0

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        app.root.mainloop()
moving_avg = target_counter[target].rolling(20).mean().fillna(0)
moving_avg.plot()
plt.show()

#print('Return to launch')
#vehicle.mode = VehicleMode("RTL")


#Close vehicle object before exiting script
print("Close vehicle object")
vehicle.close()

# Shut down simulator if it was started.
if sitl is not None:
    sitl.stop()
