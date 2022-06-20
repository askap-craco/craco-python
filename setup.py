import os
import re

from setuptools import find_packages, setup

regexp = re.compile(r'.*__version__ = [\'\"](.*?)[\'\"]', re.S)

base_package = 'craco'
base_path = os.path.dirname(__file__)

init_file = os.path.join(base_path, 'src', 'craco', '__init__.py')
with open(init_file, 'r') as f:
    module_content = f.read()

    match = regexp.match(module_content)
    if match:
        version = match.group(1)
    else:
        raise RuntimeError(
            'Cannot find __version__ in {}'.format(init_file))

with open('README.md', 'r') as f:
    readme = f.read()

with open('CHANGELOG.md', 'r') as f:
    changes = f.read()

def parse_requirements(filename):
    ''' Load requirements from a pip requirements file '''
    with open(filename, 'r') as fd:
        lines = []
        for line in fd:
            line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines

requirements = parse_requirements('requirements.txt')


if __name__ == '__main__':
    setup(
        name='craco',
        description='Python utilities for the CRACO project',
        long_description='\n\n'.join([readme, changes]),
        license='Not open source',
        url='https://github.com/strocode/craco',
        setup_requires=['pytest-runner', 'sphinx', 'recommonmark'],
        tests_require=['pytest', 'coverage', 'pytest-cov'],
        version=version,
        author='Keith Bannister',
        author_email='keith.bannister@csiro.au',
        maintainer='Keith Bannister',
        maintainer_email='keith.bannister@csiro.au',
        install_requires=requirements,
        keywords=['craco'],
        package_dir={'': 'src'},
        packages=find_packages('src'),
        zip_safe=False,
        entry_points = {
            'console_scripts': ['corrsim=craco.corrsim:_main',
                                'yaml2etcd=craco.yaml2etcd:_main',
                                'pipeline=craco.pipeline:_main',
                                'cardcap=craco.cardcap:_main',
                                'search_pipeline=craco.search_pipeline:_main',
                                'ccapfits2np=craco.ccapfits2np:_main',
                                'ifstats=craco.ifstats:_main',
            ]
        },
        classifiers=['Development Status :: 3 - Alpha',
                     'Intended Audience :: Developers',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8']
    )
