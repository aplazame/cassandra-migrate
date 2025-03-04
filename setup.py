from setuptools import setup

VERSION = '0.3.15'

install_requires = [
    "arrow==1.*",
    "cassandra-driver>=3.29.0",
    "click==8.1.7",
    "future==0.18.3",
    "python-dateutil>=2.8,<2.10",
    "PyYAML>=6,<7",
    "six==1.*",
    "tabulate==0.9.0",
    "typing-extensions>=3,<5"]

setup(name='cassandra-migrate',
      packages=['cassandra_migrate'],
      version=VERSION,
      description='Simple Cassandra database migration program.',
      long_description=open('README.rst').read(),
      url='https://github.com/Cobliteam/cassandra-migrate',
      download_url='https://github.com/Cobliteam/cassandra-migrate/archive/{}.tar.gz'.format(VERSION),
      author='Daniel Miranda',
      author_email='daniel@cobli.co',
      license='MIT',
      install_requires=install_requires,
      scripts=['bin/cassandra-migrate'],
      keywords='cassandra schema migration')
