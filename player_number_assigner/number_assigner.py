import cv2
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans

class NumberAssigner:
    def __init__(self) -> None:
        self.player_team_dict = {}
        self.player_number={}
  
    
            
    def get_classification_model(self,image):
        image_resize=cv2.resize(image,(28,56))
        image_resize=image_resize/255.0
        image2d=np.expand_dims(image_resize,axis=0)
        #image2d=image_resize.reshape(28,56,3)
        new_model = tf.keras.models.load_model('C:\\Users\\omare\\EPLTeacking\\development_and_analysis\\my_model.h5')
        self.model=new_model
        number=np.argmax(new_model.predict(image2d))+1
        return number


    
    def assign_player_number(self,frame,player_detections):
        player_numbers = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_number =  self.predict_player_number(frame,bbox)
            player_numbers.append(player_number)
    
        
    def detect_player_number(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]


        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        history=[]
        for i in clustered_image:
         for j in range(len(i)):
            if i[j]==player_cluster:
                history.append(j)
            break
        last=[]
        for i in clustered_image:
            for j in range(len(i)-1,-1,-1):
                if i[j]==player_cluster:
                    last.append(j)
            break
        count=0
        zero_value=[]
        for i in clustered_image:
            try:
                zero_arr=i[history[count]:last[count]]
                count+=1
                if np.any(zero_arr == non_player_cluster):
                    zero_value.append(zero_arr) 
            except IndexError:
                continue
            
        return zero_value

    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans
    def predict_player_number(self,frame,bbox):
        edges=self.detect_player_number(frame,bbox)
        
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0]/2),:]
        return self.get_classification_model(top_half_image)
       
    def get_player_number(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
                return self.player_team_dict[player_id]
        player_number = self.predict_player_number(frame,player_bbox)
        self.player_team_dict[player_id]=player_number
        return player_number