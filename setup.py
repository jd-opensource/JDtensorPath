from setuptools import setup, find_packages
import sys, os.path

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'jdtensorpath'))

# Environment-specific dependencies.
# extras = {
#   'atari': ['atari_py~=0.2.0', 'Pillow', 'opencv-python'],
#   'box2d': ['box2d-py~=2.3.5'],
#   'classic_control': [],
#   'mujoco': ['mujoco_py>=1.50, <2.0', 'imageio'],
#   'robotics': ['mujoco_py>=1.50, <2.0', 'imageio'],
# }

# Meta dependency groups.
# extras['all'] = [item for group in extras.values() for item in group]

setup(name='jdtensorpath',
      version=0.1,
      description='JDtensorPath is a tensor contraction path finder.',
      url='https://gitee.com/xywu1990/jd-tensor-path',
      author='Yaocheng Chen, Xingyao Wu',
      author_email='wu.x.yao@gmail.com',
      packages=[package for package in find_packages()
                if package.startswith('jdtensorpath')],
    #   zip_safe=False,
      install_requires=[
      'kahypar==1.1.6'
      ],
    #   extras_require=extras,
    #   package_data={'gym': [
    #     'envs/mujoco/assets/*.xml',
    #     'envs/classic_control/assets/*.png',
    #     'envs/robotics/assets/LICENSE.md',
    #     'envs/robotics/assets/fetch/*.xml',
    #     'envs/robotics/assets/hand/*.xml',
    #     'envs/robotics/assets/stls/fetch/*.stl',
    #     'envs/robotics/assets/stls/hand/*.stl',
    #     'envs/robotics/assets/textures/*.png']
    #   },
    #   tests_require=['pytest', 'mock'],
    #   python_requires='>=3.5',
    #   classifiers=[
    #       'Programming Language :: Python :: 3',
    #       'Programming Language :: Python :: 3.5',
    #       'Programming Language :: Python :: 3.6',
    #       'Programming Language :: Python :: 3.7',
    #       'Programming Language :: Python :: 3.8',
    #   ],
)