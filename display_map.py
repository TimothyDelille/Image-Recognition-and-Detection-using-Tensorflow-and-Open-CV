#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:21:54 2019

@author: timothydelille
"""
from tkinter import *
import math
import numpy as np
import time
import pandas as pd

def get_metres_from_deg(angle):
    return angle*6378137.0*math.pi/180.

def get_location_metres(original_location, dNorth, dEast):
    earth_radius=6378137.0 #Radius of "spherical" earth
    #Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/earth_radius

    #New position in decimal degrees
    newlat = original_location['lat'] + (dLat * 180/math.pi)
    newlon = original_location['lon'] + (dLon * 180/math.pi)
    return {'lat':newlat,'lon':newlon} #en degres
    
class Application:
    def __init__(self, mission, drone_location, waypoints, drone_altitude):
        """Constructeur de la fenêtre principale"""
        self.root =Tk()
        self.root.title('Vue de dessus de la map')
        self.width = 600
        self.height = 600
        self.canvas = Canvas(self.root, width=str(self.width), height=str(self.height))
        self.canvas.pack(side =TOP, padx =5, pady =5)
        self.draw_map(mission, waypoints)
        self.draw_drone(drone_location, self.factor, waypoints, self.waypoints, drone_altitude)
        self.draw_altitude(drone_altitude)

    def draw_map(self, mission, waypoints):
        R = 6378137.0 #Radius of earth
        if mission=='single' or mission=='selected':
            
            dlat = (waypoints[1]['lat']-waypoints[0]['lat'])*math.pi/180.
            dlon = (waypoints[1]['lon']-waypoints[0]['lon'])*math.pi/180.

            dy = dlat*R
            dx = dlon*R
            
            if abs(dx)>=abs(dy):
                #0.8*self.width/dx = y_pixels/dy => y_pixels = dy*factor
                #facteur entre les pixels et la réalite -> en pixels/m
                factor = 0.8*float(self.width)/dx
                if dx >= 0:
                    x0 = 0.1*self.width
                    x1 = 0.9*self.width
                elif dx<0:
                    x1 = 0.1*self.width
                    x0 = 0.9*self.width
                if dy >= 0:
                    y1 = (self.height-dy*factor)/2.
                    y0 = y1 + dy*factor
                elif dy < 0:
                    y0 = (self.height-dy*factor)/2.
                    y1 = y0 + dy*factor
            elif abs(dy)>abs(dx):
                factor = 0.8*float(self.height)/dy
                if dy >= 0:
                    y1 = 0.1*self.height
                    y0 = 0.9*self.height
                elif dy<0:
                    y0 = 0.1*self.height
                    y1 = 0.9*self.height
                if dx >= 0:
                    x0 = (self.width-dx*factor)/2.
                    x1 = x0 + dx*factor
                elif dx < 0:
                    x1 = (self.width-dx*factor)/2.
                    x0 = x1 + dx*factor
            
            self.factor = factor
            self.origin = (x0,y0)
            self.waypoints = [[x0,y0],[x1,y1]]
            #self.canvas.delete('all')
            self.canvas.create_line(x0, y0, x1, y1)
            self.canvas.create_text(x0, y0, text='takeoff (0,0)', font='calibri 8 italic')
            self.canvas.create_text(x1, y1, text='landing ('+ str(round((x1-x0)/self.factor,2)) + ',' + str(round((y0-y1)/self.factor,2)) + ')', font='calibri 8 italic', anchor=S)
            
        elif mission=='full':
            dx = np.zeros((4,4))
            dy = np.zeros((4,4))
            for i in range(4):
                for j in range(4):
                    dy[i][j] = (waypoints[j]['lat']-waypoints[i]['lat'])*R*math.pi/180.
                    dx[i][j] = (waypoints[j]['lon']-waypoints[i]['lon'])*R*math.pi/180.
            
            max_dx = np.unravel_index(np.argmax(dx, axis=None), dx.shape)
            max_dy = np.unravel_index(np.argmax(dy, axis=None), dy.shape)
            
            
            #factor en pixels/m -> rapport entre la taille d'un pixel et la réalité
            #doit être le même pour l'axe x et l'axe y
            factor = min(0.6*float(self.width)/dx[max_dx],0.6*float(self.height)/dy[max_dy])
            self.factor = factor
            #x_1 - x_0 = dx01*factor
            #width - x_1 = x_0
            #d'ou x_0 = width - (dx01*factor+width)//2          
            x = np.zeros(4)
            x[max_dx[0]]=(self.width-dx[max_dx]*factor)//2
            for i in range(len(x)):
                x[i] = dx[max_dx[0]][i]*factor+x[max_dx[0]]
            
            y = np.zeros(4)
            y[max_dy[1]]=(self.height-dy[max_dy]*factor)//2
            for j in range(len(y)):
                y[j] = y[max_dy[1]] + dy[j][max_dy[1]]*factor
            
            
            x0,x1,x2,x3 = x.astype(int)
            y0,y1,y2,y3 = y.astype(int)
            self.origin = (x0,y0)
                
            self.waypoints = [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
            self.canvas.create_polygon(x0, y0, x1, y1, x2, y2, x3, y3, x0, y0, outline='gray', fill='white', width=1)
            self.canvas.create_text(x0, y0, text='1 (0,0)', font='calibri 8 italic')
            self.canvas.create_text(x1, y1, text='2 ('+ str(round((x1-x0)/self.factor,2)) + ',' + str(round((y0-y1)/self.factor,2)) + ')', font='calibri 8 italic', anchor=S)
            self.canvas.create_text(x2, y2, text='3 ('+ str(round((x2-x0)/self.factor,2)) + ',' + str(round((y0-y2)/self.factor,2)) + ')', font='calibri 8 italic',anchor=S)
            self.canvas.create_text(x3, y3, text='4 ('+ str(round((x3-x0)/self.factor,2)) + ',' + str(round((y0-y3)/self.factor,2)) + ')', font='calibri 8 italic')
                
    def get_distance(self, aLocation1, aLocation2):
        dlat = aLocation2['lat'] - aLocation1['lat']
        dlong = aLocation2['lon'] - aLocation1['lon']
        return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5
          
    def draw_drone(self, drone_location, factor, waypoints, pixel_waypoints, drone_altitude):
        R = 6378137.0
        dlat = (drone_location['lat']-waypoints[0]['lat'])*math.pi/180.
        dlon = (drone_location['lon']-waypoints[0]['lon'])*math.pi/180.
        dx = dlon*R
        dy = dlat*R
        x = pixel_waypoints[0][0] + dx*factor
        y = pixel_waypoints[0][1] - dy*factor
        r=10
        self.drone = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='grey')
        d_recogn = drone_altitude*0.2*factor
        self.d_recogn = self.canvas.create_oval(x-d_recogn//2, y-d_recogn//2, x+d_recogn//2, y+d_recogn//2)
        self.drone_position = np.array([x,y])
        x0,y0 = self.origin
        self.drone_text = self.canvas.create_text(x,y-10, text='(' + str(round((x-x0)/self.factor,2)) + ',' + str(round((y0-y)/self.factor,2)) + ')', anchor=S, font='calibri 8 italic')
    
    def draw_altitude(self, drone_altitude, max_alt = 20):
        x0,y0,x1,y1 = [30,10,30,100] #normalement x0 = x1
        self.canvas.create_line(x0,y0,x1,y1)
        self.canvas.create_line(x1,y1,x1+50,y1) #ligne du sol
        self.canvas.create_text(x1,y1,text='0m', font='calibri 8', anchor=E)
        self.canvas.create_text(x0,y0,text=str(max_alt)+'m', font='calibri 8', anchor=E)
        factor = (y1-y0)/max_alt #en px/m
        alt_px = y1 - drone_altitude*factor
        self.drone_alt_text = self.canvas.create_text(x0, alt_px, text=str(round(drone_altitude,2))+'m', font='calibri 8', anchor=SE) 
        self.drone_alt_drone = self.canvas.create_rectangle(x0+10,alt_px,x0+40,alt_px-10,fill='grey')
        
    def move_drone(self, old_location, new_location, factor, drone, drone_altitude):
        R = 6378137.0
        dlat = (new_location['lat']-old_location['lat'])*math.pi/180.
        dlon = (new_location['lon']-old_location['lon'])*math.pi/180.
        dx = dlon*R*factor
        dy = -dlat*R*factor
        self.drone_position = self.drone_position + np.array([dx,dy])
        self.canvas.move(drone, dx, dy)
        d_recogn = drone_altitude*0.2*factor
        self.canvas.delete(self.d_recogn)
        x,y = self.drone_position
        self.d_recogn = self.canvas.create_oval(x-d_recogn//2, y-d_recogn//2, x+d_recogn//2, y+d_recogn//2)
        
        self.canvas.delete(self.drone_text)
        x0,y0 = self.origin
        x,y = self.drone_position
        self.drone_text = self.canvas.create_text(x,y-10, text='(' + str(round((x-x0)/self.factor,2)) + ',' + str(round((y0-y)/self.factor,2)) + ')', anchor=S, font='calibri 8 italic')
        
        self.canvas.delete(self.drone_alt_text)
        self.canvas.delete(self.drone_alt_drone)
        self.draw_altitude(drone_altitude)
        self.root.update()
        
    def draw_cross(self, color, dlocation, factor, drone_position):
        w = 4
        l = 20
        x = drone_position[0] + dlocation[0]*factor
        y = drone_position[1] - dlocation[1]*factor 
        self.canvas.create_polygon([x-l/2,y+w/2, 
                                    x-l/2,y-w/2,
                                    x-w/2,y-w/2,
                                    x-w/2,y-l/2,
                                    x+w/2,y-l/2,
                                    x+w/2,y-w/2,
                                    x+l/2,y-w/2,
                                    x+l/2,y+w/2,
                                    x+w/2,y+w/2,
                                    x+w/2,y+l/2,
                                    x-w/2,y+l/2,
                                    x-w/2,y+w/2], fill=color)
        x0,y0=self.origin
        self.canvas.create_text(x,y, text='(' + str(round((x-x0)/self.factor,2)) + ',' + str(round((y0-y)/self.factor,2)) + ')', anchor=CENTER, font='calibri 8 italic')
    
    def draw_rectangle(self, color, dlocation, factor, drone_position):
        w = 4
        l = 20
        x = drone_position[0] + dlocation[0]*factor
        y = drone_position[1] - dlocation[1]*factor 
        self.canvas.create_rectangle(x-l/2,y+w/2,
                                     x+l/2,y-w/2, fill=color)
        x0,y0=self.origin
        self.canvas.create_text(x,y, text='(' + str(round((x-x0)/self.factor,2)) + ',' + str(round((y0-y)/self.factor,2)) + ')', anchor=CENTER, font='calibri 8 italic')
        
    def capture(self, filepath):
        self.canvas.postscript(file=filepath+".ps", colormode='color')
        
if __name__=='__main__':
    drone_location = {'lat':48.856613, 'lon':2.352222}
    #get_location_metres(location, dNorth, dEast) Lat->dNorth, Lon->dEast
    waypoints = [get_location_metres(drone_location,0,0),
                 get_location_metres(drone_location,3,11),
                 get_location_metres(drone_location,19,13),
                 get_location_metres(drone_location,13,3)]
    
    drone_altitude=10
    d_recogn = 0.2*drone_altitude #2m
    
    area_recogn = 0.25*np.pi*d_recogn**2
    segments = [] #les 4 segments qui composent l'aire a explorer: un couple [dx,dy] par segment
    norms = [] #les normes de ces segments
    convert = 6378137.0*math.pi/180.
    
    for i in range(4):
        segments.append([(waypoints[(i+1)%4]['lon']-waypoints[i%4]['lon'])*convert,(waypoints[(i+1)%4]['lat']-waypoints[i%4]['lat'])*convert])
        norms.append(np.linalg.norm(segments[-1]))
        
    N = np.zeros(2)
    N[0] = int(max(norms[0],norms[2])) #segments 0 et 2
    N[1] = int(max(norms[1],norms[3])) #segments 1 et 3
    N = N/d_recogn
    N = N.astype(int)
    
    points_segment_0 = []
    points_segment_1 = []
    points_segment_2 = []
    points_segment_3 = []
    
    u = [np.array(segments[0])/N[0],np.array(segments[1])/N[1],np.array(segments[2])/N[0],np.array(segments[3])/N[1]]
    #l'origine et le point 0
    for i in range(N[1]):
        d = (i+0.5)
        points_segment_3.append(list(-np.array(u[3])*d))
        points_segment_1.append(list(np.array(segments[0])+d*np.array(u[1])))
    
    for i in range(N[0]):
        d = (i+0.5)
        points_segment_0.append(list(np.array(u[0])*d))
        points_segment_2.append(list(-np.array(segments[3])-d*np.array(u[2])))
            
    intercept = []
    slopes = []
    cpt=0
    old_location = drone_location
    
    app = Application('full',old_location,waypoints,drone_altitude)

    for i in range(N[1]):
        x0 = app.origin[0]+points_segment_3[i][0]*app.factor
        y0 = app.origin[1]-points_segment_3[i][1]*app.factor
        x1 = app.origin[0]+points_segment_1[i][0]*app.factor
        y1 = app.origin[1]-points_segment_1[i][1]*app.factor
        #app.canvas.create_line(x0, y0, x1, y1)
    for i in range(N[0]):
        x0 = app.origin[0]+points_segment_0[i][0]*app.factor
        y0 = app.origin[1]-points_segment_0[i][1]*app.factor
        x1 = app.origin[0]+points_segment_2[i][0]*app.factor
        y1 = app.origin[1]-points_segment_2[i][1]*app.factor
        #app.canvas.create_line(x0, y0, x1, y1)
    
    x1,y1 = np.array(u[2])*app.factor
    x1 = app.origin[0]+x1
    y1 = app.origin[1]-y1
    #app.canvas.create_line(app.origin[0], app.origin[1], x1, y1, width=3)
    
    lines_02 = [segments[3]]#toutes les lignes entre les segments 0 et 2
    x_y = np.array(points_segment_2)-np.array(points_segment_0)
    lines_02 = lines_02 + x_y.tolist()
    lines_02.append(segments[1])
    
    lines_31 = [segments[0]]
    x_y = np.array(points_segment_1)-np.array(points_segment_3)
    lines_31 = lines_31 + x_y.tolist()
    lines_31.append(segments[2])
    
    slopes_02 = []
    slopes_31 = []
    for i in range(len(lines_02)):
        if lines_02[i][0]==0:
            slopes_02.append('inf')
        else:
            slopes_02.append(np.array(lines_02).T[1][i]/np.array(lines_02).T[0][i])
    for i in range(len(lines_31)):
        if lines_31[i][0]==0:
            slopes_31.append('inf')
        else:
            slopes_31.append(np.array(lines_31).T[1][i]/np.array(lines_31).T[0][i])

    
    points_02 = [[0,0]] + points_segment_0 + [[(waypoints[1]['lon']-waypoints[0]['lon'])*convert, (waypoints[1]['lat']-waypoints[0]['lat'])*convert]]
    points_31 = [[0,0]] + points_segment_3 + [[(waypoints[3]['lon']-waypoints[0]['lon'])*convert, (waypoints[3]['lat']-waypoints[0]['lat'])*convert]]
    intercept_02 = []
    intercept_31 = []
    for i in range(len(slopes_02)):
        if slopes_02[i]=='inf':
            intercept_02.append('inf')
        else:
            intercept_02.append(np.array(points_02).T[1][i]-slopes_02[i]*np.array(points_02).T[0][i]) #b = y - ax
    for i in range(len(slopes_31)):
        if slopes_31[i]=='inf':
            intercept_31.append('inf')
        else:
            intercept_31.append(np.array(points_31).T[1][i]-slopes_31[i]*np.array(points_31).T[0][i])
    
    points=[]
    for i in range(len(lines_02)):
        d = intercept_02[i]
        c = slopes_02[i]
        x=[]
        y=[]
        for j in range(len(lines_31)):
            if c == 'inf':
                x.append(points_02[i][0])
                y.append(slopes_31[j]*x[-1]+intercept_31[j])
            elif slopes_31[j]=='inf': 
                x.append(points_31[i][0])
                y.append(slopes_02[i]*x[-1]+intercept_02[i])
            else:
                x.append((d-intercept_31[j])/(slopes_31[j]-c))
                y.append(slopes_02[i]*x[-1]+ intercept_02[i])
        x_y = np.array([x,y]).T.tolist()
        points.append(x_y)
        
    for i in points:
        for j in i:
            x,y=j
            x=app.origin[0]+x*app.factor
            y=app.origin[1]-y*app.factor
            app.canvas.create_oval(x,y,x+1,y+1)
            
    trajectory = [points[0][0],points[0][1]] #premier decalage
    #Oon doit faire 2*min(N) segments
    for k in range(0,2*min(N)):
        i = 1+k//4
        if k%4==0:
            trajectory.append(points[-i-1][i])
        elif k%4==1:
            trajectory.append(points[-i-1][-i-1])
        elif k%4==2:
            trajectory.append(points[i][-i-1])
        elif k%4==3:
            trajectory.append(points[i][i+1])
    trajectory.append(trajectory[0]) #retour au départ
    
    cmds_xy = np.array([waypoints[0]['lon']*convert,waypoints[0]['lat']*convert])+np.array(trajectory)
    cmds_lat_lon = np.array([cmds_xy.T[0],cmds_xy.T[1]]).T/convert
    cmds_lat_lon = cmds_lat_lon.tolist()
    
    for i in range(len(trajectory)-1):
        x0 = app.origin[0]+trajectory[i][0]*app.factor
        y0 = app.origin[1]-trajectory[i][1]*app.factor
        x1 = app.origin[0]+trajectory[i+1][0]*app.factor
        y1 = app.origin[1]-trajectory[i+1][1]*app.factor
        app.canvas.create_line(x0,y0,x1,y1)
            
    old_traj = [0,0]
    for i in range(len(trajectory)):
        new_location = get_location_metres(old_location,trajectory[i][1]-old_traj[1],trajectory[i][0]-old_traj[0])
        app.move_drone(old_location, new_location, app.factor, app.drone, drone_altitude)
        old_location = new_location
        old_traj = trajectory[i]
        time.sleep(0.25)
        #if cpt==-1:
        #    app.capture("./capture_map")
       # cpt+=1
    
    app.root.mainloop()