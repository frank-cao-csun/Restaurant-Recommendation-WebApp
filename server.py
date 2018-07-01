import os, cherrypy
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from paste.translogger import TransLogger
from app import create_app

def init_spark_session():
    #load the spark session(local mode)
    ss = (SparkSession.builder
         .master("local")
         .appName("Restaurant Recommendations")
         .getOrCreate())

    ss.sparkContext.setLogLevel("WARN")
    ss.sparkContext.addPyFile('engine.py')
    ss.sparkContext.addPyFile('app.py')
   
    return ss

def run_server(app):
    # Enable WSGI access logging via Paste
    app_logged = TransLogger(app)
 
    # Mount the WSGI callable object (app) on the root directory
    cherrypy.tree.graft(app_logged, '/')
 
    # Set the configuration of the web server
    cherrypy.config.update({
        'engine.autoreload.on': True,
        'log.screen': True,
        'server.socket_port': 8080,
        'server.socket_host': '0.0.0.0'
    })
 
    # Start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()

if __name__ == '__main__':

    ss=init_spark_session()
    dataset_path=os.path.join('dataset','json')
    app=create_app(ss,dataset_path)
    #start the web server
    run_server(app);
    
