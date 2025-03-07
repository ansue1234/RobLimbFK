from setuptools import find_packages, setup
from glob import glob
import os
package_name = 'model_runner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ansue1234',
    maintainer_email='ansue1234@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pred = model_runner.fk_predictor:main',
            'looper=model_runner.open_loop_input_traj:main',
            'filter=model_runner.filter:main',
            'visualizer=model_runner.visualizer:main',
            'plotter=model_runner.cvs_plotter:main',
            'broadcaster=model_runner.broadcaster:main',
            'policy=model_runner.policy:main',
            'feeder=model_runner.waypoint_feeder:main',
            'pid=model_runner.pid:main',
        ],
    },
)
