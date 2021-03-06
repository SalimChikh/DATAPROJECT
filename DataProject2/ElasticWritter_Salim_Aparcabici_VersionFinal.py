#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:36:13 2020

@author: edem
"""

from __future__ import absolute_import

import argparse
import logging

import apache_beam as beam

from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import StandardOptions

from apache_beam.options.pipeline_options import SetupOptions

from elasticsearch import Elasticsearch 

import json
import utm


class ConvertUTM(beam.DoFn):
    """
    Filter data for inserts
    """

    def process(self, element):
       
        #{"empty_slots":17,"extra":{"address":"Economista Gay - Constituci\xc3\xb3n","banking":false,"bonus":false,"last_update":1578482815000,"slots":20,"status":"OPEN","uid":136},"free_bikes":3,"id":"1f6b81722ca23ce520f77207b868afa9","latitude":39.4899091610835,"longitude":-0.375701108044157,"name":"136_CALLE_ECONOMISTA_GAY","timestamp":"2020-01-08T11:34:20.782000Z"}'
       
        item = json.loads(element)
        
        coords = item['geometry']
        huso = 30
        X = coords['coordinates'][0]
        Y = coords['coordinates'][1]
        lat,lon = utm.to_latlon(X, Y, huso, 'S')
        
        properties=item['properties']
                
        return [{'type':item['type'],
                 'plazas':properties['plazas'],
                 'tipo':properties['tipo'],
                 'id':properties['id'],
                 'typefeature':coords['type'],
                 'coordinates':str(lat)+","+str(lon)   
                 }]
#class LocationConcat(beam.DoFn):
    #"""
    #Filter data for inserts
    #"""

 #   def process(self, element):
        
        #{"empty_slots":17,"extra":{"address":"Economista Gay - Constituci\xc3\xb3n","banking":false,"bonus":false,"last_update":1578482815000,"slots":20,"status":"OPEN","uid":136},"free_bikes":3,"id":"1f6b81722ca23ce520f77207b868afa9","latitude":39.4899091610835,"longitude":-0.375701108044157,"name":"136_CALLE_ECONOMISTA_GAY","timestamp":"2020-01-08T11:34:20.782000Z"}'
        
       # item = json.loads(element)
        #return [{'empty_slots':item['empty_slots'],
         #        'free_bikes':item['free_bikes'],
          #       'id':item['id'],
           #      'name':item['name'],
            #     'timestamp':item['timestamp'],
             #    'latitude':item['latitude'],
              #   'longitude':item['longitude'],
               #  'address':item['extra']['address'],
                # 'banking':item['extra']['banking'],
                 #'bonus':item['extra']['bonus'],
                 #'status':item['extra']['status'],
                 #'uid':item['extra']['uid'],
                 #'location':str(item['latitude'])+","+str(item['longitude'])                
                 #}]


class IndexDocument(beam.DoFn):
   
    es=Elasticsearch([{'host':'localhost','port':9200}])
    
    def process(self,element):
        
        res = self.es.index(index='parking4',body=element)
        
        print(res)
 
        
    
def run(argv=None, save_main_session=True):
  """Main entry point; defines and runs the wordcount pipeline."""
  parser = argparse.ArgumentParser()
  
  #1 Replace your hackathon-edem with your project id 
  parser.add_argument('--input_streaming',
                      dest='input_streaming',
                      #1 Add your project Id and topic name you created
                      # Example projects/versatile-gist-251107/topics/iexCloud',
                      default='projects/hackaton-salim/topics/parking4',
                      help='Input file to process.')
  #2 Replace your hackathon-edem with your project id 
  parser.add_argument('--input_parkstreaming',
                      dest='input_parkstreaming',
                      #3 Add your project Id and Subscription you created you created
                      # Example projects/versatile-gist-251107/subscriptions/quotesConsumer',
                      default='projects/hackaton-salim/subscriptions/parkingstream4',
                      help='Input subscription Parking')
  
  
  
    
  known_args, pipeline_args = parser.parse_known_args(argv)

  # We use the save_main_session option because one or more DoFn's in this
  # workflow rely on global context (e.g., a module imported at module level).
  pipeline_options = PipelineOptions(pipeline_args)
   
  google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
  #3 Replace your hackathon-edem with your project id 
  google_cloud_options.project = 'hackathon-salim'
  google_cloud_options.job_name = 'myjob'
 
  # Uncomment below and add your bucket if you want to execute on Dataflow
  #google_cloud_options.staging_location = 'gs://edem-bucket-roberto/binaries'
  #google_cloud_options.temp_location = 'gs://edem-bucket-roberto/temp'

  pipeline_options.view_as(StandardOptions).runner = 'DirectRunner'
  #pipeline_options.view_as(StandardOptions).runner = 'DataflowRunner'
  pipeline_options.view_as(StandardOptions).streaming = True

 
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session


 

  p = beam.Pipeline(options=pipeline_options)


  # Read the pubsub messages into a PCollection.
  parking4 = p | beam.io.ReadFromPubSub(subscription=known_args.input_parkstreaming)

  # Print messages received
 
  
  
  parking4 = ( parking4 | beam.ParDo(ConvertUTM()))
  
  parking4 | 'Print Quote' >> beam.Map(print)
  
  # Store messages on elastic
  parking4 | 'Bici Parking' >> beam.ParDo(IndexDocument())
  
  
  
 
  result = p.run()
  result.wait_until_finish()

  
if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()