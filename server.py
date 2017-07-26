import os, cherrypy
from pyspark import SparkContext, SparkConf
from paste.translogger import TransLogger
from app import create_app


def init_spark_context():
    #load the spark context
    conf = SparkConf().setAppName("Restaurant Recommendations")
    #Adding additional Python modules
    sc=SparkContext(conf=conf, pyFiles=['engine.py','app.py'])
    return sc

def run_server(app):
    # Enable WSGI access logging via Paste
    app_logged = TransLogger(app)
 
    # Mount the WSGI callable object (app) on the root directory
    cherrypy.tree.graft(app_logged, '/')
 
    # Set the configuration of the web server
    cherrypy.config.update({
        'engine.autoreload.on': True,
        'log.screen': True,
        'server.socket_port': 5432,
        'server.socket_host': '0.0.0.0'
    })
 
    # Start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()

if __name__ == '__main__':
    sc=init_spark_context()
    dataset_path=os.path.join('dataset','Yelp')
    app=create_app(sc,dataset_path)
    
    #start the web server
    run_server(app);
    
