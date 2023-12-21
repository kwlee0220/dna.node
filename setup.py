from setuptools import setup, find_packages
import dna

setup( 
    name = 'dna.node',
    version = dna.__version__,
    description = 'DNA framework',
    author = 'Kang-Woo Lee',
    author_email = 'kwlee@etri.re.kr',
    url = 'https://github.com/kwlee0220/dna.node',
	entry_points={
		'console_scripts': [
			'dna_node_server = scripts.dna_node_server:main',
			'dna_node = scripts.dna_node:main',
			'dna_track = scripts.dna_detect:main',
			'dna_detect = scripts.dna_detect:main',
			'dna_show = scripts.dna_show:main',
   
            # 
            # MCMOT relateds
            #
            'dna_show_gtracks = scripts.dna_show_global_tracks:main',
            'dna_smooth = scripts.dna_smooth_trajs:main',
   
            # 
            # Supporting tools
            # 
			'dna_replay = scripts.dna_replay_node_events:main',
            'dna_merge_sort_events = scripts.dna_merge_sort_events:main',
			'dna_show_mc_locations = scripts.dna_show_mc_locations:main',
            'dna_print_events = scripts.dna_print_events:main',
			'dna_export = scripts.dna_export_topic:main',
			'dna_show_multiple_videos = scripts.dna_show_multiple_videos:main',
            
			# 'dna_download = scripts.dna_download_node_events:main',
			# 'dna_import = scripts.dna_import_topic:main',
			# 'dna_draw_trajs = scripts.dna_draw_trajs:main',
		],
	},
    install_requires = [
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',

        'opencv-python>=4.1.2',
        'kafka-python',
        
        # ffmpeg library
        'ffmpeg-python',

        'omegaconf>=2.1.2',
        'tqdm>=4.41.0',
        'Shapely',
        'pyyaml',
        'gdown',
        
        # parse date-time string
        'python-dateutil',

        # geodesic transformation
        'pyproj',
        
        # redis
        'redis',

        # yolov5
        'ipython',
        'ultralytics',
        # 'psutil',
        # # ultralytics-yolov5
        # 'gitpython',
        # 'gitdb',
        # 'smmap',
        
        # protobuf
        'protobuf',
    ],
    packages = find_packages(),
    package_dir={'conf': 'conf'},
    package_data = {
        'conf': ['logger.yaml']
    },
    python_requires = '>=3.10',
    zip_safe = False
)
