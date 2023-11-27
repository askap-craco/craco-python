import os
import re
import glob

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
        scripts=glob.glob('scripts/*'),
        entry_points = {
            'console_scripts': ['corrsim=craco.corrsim:_main',
                                'yaml2etcd=craco.yaml2etcd:_main',
                                'pipeline=craco.pipeline:_main',
                                'cardcap=craco.cardcap:_main',
                                'search_pipeline=craco.search_pipeline:_main',
                                'ccapfits2np=craco.ccapfits2np:_main',
                                'ifstats=craco.ifstats:_main',
                                'ccapmerger=craco.cardcapmerger:_main',
                                'metadatafile=craco.metadatafile:_main',
                                'ccapfits2uvfits=craco.ccapfits2uvfits:_main',
                                'linkdirs=craco.linkdirs:_main',
                                'ccapfits2fil=craco.ccap2fil:_main',
                                'ccaphdr=craco.ccaphdr:_main',
                                'plot_cardcap=craco.plot_cardcap:_main',
                                'plot_calibration=craco.calibration:_main',
                                'plot_cand=craco.plot_cand:_main',
                                'switchconnections=craco.switchconnections:_main',
                                'mpipipeline=craco.mpipipeline:_main',
                                'mpi_transpose_test=craco.mpi_transpose_test:_main',
                                'test_card_averager=craco.test_card_averager:_main',
                                'obsman=craco.obsman:_main',
                                'savescan=craco.savescan:_main',
                                'calibration=craco.calibration:_main',
                                'dusb=craco.dusb:_main',
                                'prepsb=craco.prepsb:_main',
                                'prthd=craco.prthd:_main',
                                'plotpcap=craco.plotpcap:_main',
                                'metadatasaver=craco.obsman.metadatasaver:_main',
                                'fixuvfits=craco.fixuvfits:_main',
                                'pltrescale=craco.pltrescale:_main',
                                'pltccap=craco.pltccap:_main',
                                'tab_filterbank=craco.tab_filterbank:_main',
                                'pretty_imager=craco.pretty_imager:main',
                                'candpipe=craco.candpipe.candpipe:_main',
                                'mpicalibrate=craco.mpicalibrate:_main',
                                "cracoslackbot=craco.craco_slackbot:_main",
                                'calibrate_vis=craco.calibrate_vis:main',
                                'sky_subtract_vis=craco.sky_subtract_vis:main',
                                'dedisperse_vis=craco.dedisperse_vis:main',
                                'attach_uvws_uvfits=craco.attach_uvws_uvfits:main',
                                
            ],
        },
        classifiers=['Development Status :: 3 - Alpha',
                     'Intended Audience :: Developers',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8']
    )
