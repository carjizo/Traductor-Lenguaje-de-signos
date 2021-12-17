import cv2
import numpy as np
import imutils
import math

cap = cv2.VideoCapture(0)

bg = None # fondo

# COLORES PARA VISUALIZACIÓN
color_start = (204,204,0)
color_end = (204,0,204)
color_far = (255,0,0)
color_start_far = (204,204,0)
color_far_end = (204,0,204)
color_start_end = (0,255,255)
color_contorno = (0,255,0)
color_ymin = (0,130,255) # Punto más alto del contorno
color_angulo = (0,255,255)
color_d = (0,255,255)
color_fingers = (0,255,255)

while True:
    ret, frame = cap.read()
    if ret == False: break

    # Redimensionar la imagen para que tenga un ancho de 640
    frame = imutils.resize(frame,width=640)
    #cv2.imshow('frame1',frame)
    frame = cv2.flip(frame,1) #efecto espejo
    #cv2.imshow('frame espejo',frame)
    frameAux = frame.copy()
    Region_Interes = frame[0:370,360:620]
    Dibujo = np.zeros(Region_Interes.shape, np.uint8)

    if bg is not None:
        

        # Determinar la región de interés
        ROI = frame[50:350,380:600]
        cv2.rectangle(frame,(380-2,50-2),(600+2,350+2),color_fingers,1)
        
        #cv2.rectangle(frame,(380-2,50-2),(600+2,300+2),color_fingers,1)
       
        grayROI = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
        #cv2.imshow('ROI',ROI)
        #cv2.imshow('grayROI',grayROI)

        # Región de interés del fondo de la imagen
        bgROI = bg[50:350,380:600]
        #cv2.imshow('bgROI',bgROI)
        # Determinar la imagen binaria (background vs foreground)
        dif = cv2.absdiff(grayROI, bgROI)
        #cv2.imshow('dif',dif)
        _, th1 = cv2.threshold(dif, 30, 255, cv2.THRESH_BINARY)
        th = cv2.medianBlur(th1, 7)
        #cv2.imshow('th',th)

        # Encontrando los contornos de la imagen binaria
        cnts, _ = cv2.findContours(th,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:1]
        
        #cv2.drawContours(ROI,cnts,0, (0,255,0),2)

        

        for cnt in cnts:
            
            area_of_contour = cv2.contourArea(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(ROI,(x,y),(x+w,y+h),(0,0,255),1) # Red Rectangle 
            

            # Contorno encontrado a través de cv2.convexHull
            hull1 = cv2.convexHull(cnt)
            cv2.drawContours(ROI,[hull1],0,color_contorno,2)
            drawing = np.zeros(ROI.shape,np.uint8)

            cv2.drawContours(drawing,[cnt],0,(0,255,0),2)
            cv2.drawContours(drawing,[hull1],0,(0,255,255),2)
            hull2 = cv2.convexHull(cnt,returnPoints=False)
            defects = cv2.convexityDefects(cnt,hull2)
            
            count_defects = 0
            cv2.drawContours(th1, cnts, -1, (0,255,0), 3)
            
            if defects is not None:

                inicio = [] # Contenedor en donde se almacenarán los puntos iniciales de los defectos convexos
                fin = [] # Contenedor en donde se almacenarán los puntos finales de los defectos convexos
                fingers = 0 # Contador para el número de dedos levantados
                
                for i in range(defects.shape[0]):

                    s,e,f,d = defects[i,0]
                    start = cnt[s][0]
                    end = cnt[e][0]
                    far = cnt[f][0]
                    
                    # Encontrar el triángulo asociado a cada defecto convexo para determinar ángulo					
                    a = np.linalg.norm(far-end)
                    b = np.linalg.norm(far-start)
                    c = np.linalg.norm(start-end)

                    angulo = np.arccos((np.power(a,2)+np.power(b,2)-np.power(c,2))/(2*a*b))
                    angulo = np.degrees(angulo)
                    angulo = int(angulo)
                    cv2.circle(ROI,tuple(far),4,(0,0,255),-1)
                    
                    if angulo <= 90:
                        count_defects += 1
                    cv2.line(ROI,tuple(start),tuple(end),(0,255,0),3)
                        
                moment = cv2.moments(cnt)   
                perimeter = cv2.arcLength(cnt,True)
                area = cv2.contourArea(cnt)
                
                (x,y),radius = cv2.minEnclosingCircle(cnt)
                center = (int(x),int(y))
                radius = int(radius)
                cv2.circle(ROI,center,radius,(255,0,0),2)
                
                area_of_circle= np.pi*radius*radius
                
                hull_test = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull_test)
                solidity = float(area)/hull_area
                
                aspect_ratio = float(w)/h
                
                rect_area = w*h
                extent = float(area)/rect_area

                (x,y),(MA,ma),angle_t = cv2.fitEllipse(cnt)
                

                if count_defects == 0 :
                    if 0.81 < solidity < 0.85 :      
                        cv2.putText(frame,'U',(390,45), 1, 4,(0,0,255),2,cv2.LINE_AA)

                    elif 0.60 < solidity < 0.75 :
                        if angle_t >120 :
                            cv2.putText(frame,'L',(390,45), 1, 4,(0,0,255),2,cv2.LINE_AA)
                        else :
                            cv2.putText(frame,'Y',(390,45), 1, 4,(0,0,255),2,cv2.LINE_AA)
                    elif 0.8< solidity < 0.99 :
                        if angle_t > 160:
                            if extent > 0.61:
                                cv2.putText(frame,'B',(390,45), 1, 4,(0,0,255),2,cv2.LINE_AA)
                            else:
                                cv2.putText(frame,'R',(390,45), 1, 4,(0,0,255),2,cv2.LINE_AA)
                    
                        else:
                            cv2.putText(frame,'N',(390,45), 1, 4,(0,0,255),2,cv2.LINE_AA)
                    
                


                elif count_defects ==1:

                    if 180>angle_t > 150:

                        cv2.putText(frame, "V", (390,45), 1, 4,(0,0,255),2,cv2.LINE_AA) 

                    elif 120 < angle_t < 132:

                        cv2.putText(frame, "C", (390,45), 1, 4,(0,0,255),2,cv2.LINE_AA)
                        
                    elif 20 < angle_t < 50:

                        cv2.putText(frame, "P", (390,45), 1, 4,(0,0,255),2,cv2.LINE_AA)

                        
                elif count_defects ==2:
                    if solidity > 0.7:
                        cv2.putText(frame, "W", (390,45), 1, 4,(0,0,255),2,cv2.LINE_AA)
                    else:
                        cv2.putText(frame, "F", (390,45), 1, 4,(0,0,255),2,cv2.LINE_AA)


                
            cv2.imshow('th',th)
            cv2.imshow('drawing',drawing)
    #cv2.imshow('Frame',frame)
    cv2.imshow('Video de Entrada',Region_Interes)

    k = cv2.waitKey(10)
    
    if k == ord('i'):
        bg = cv2.cvtColor(frameAux,cv2.COLOR_BGR2GRAY)
    if k == ord('s'):
        print(count_defects)
        print(area)
        print(perimeter)
        print(angle_t) 
        print(solidity) 
        print(extent)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()