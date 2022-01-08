import os
import json
from osmaug.osm.getosm_main import getosm_function
from osmaug.transformation.augmosm_main import augmosm_function
from osmaug.combine.collosm_main import collosm_function


def main():
    current_dir = os.path.abspath(os.getcwd())
    config_dir = os.path.join(current_dir, 'config')
    f_config = open(os.path.join(config_dir, 'osm_z17.json'), encoding='utf8')
    params = json.load(f_config)
    params['resdir'] = os.path.join(current_dir, 'data', 'tile')
    params['driver_bin'] = "C:/Users/vemun/PycharmProjects/Master/osmaug/geckodriver.exe"
    print(getosm_function(params))


def main_bus():
    current_dir = os.path.abspath(os.getcwd())
    config_dir = os.path.join(current_dir, 'config')
    f_config = open(os.path.join(config_dir, 'b_adt_wms_z17.json'), encoding='utf8')
    params = json.load(f_config)
    params['resdir'] = os.path.join(current_dir, 'data', 'tile')
    params['driver_bin'] = "C:/Users/vemun/PycharmProjects/Master/osmaug/geckodriver.exe"
    getosm_function(params)


def main_heavy():
    current_dir = os.path.abspath(os.getcwd())
    config_dir = os.path.join(current_dir, 'config')
    f_config = open(os.path.join(config_dir, 'h_abt_wms_z17.json'), encoding='utf8')
    params = json.load(f_config)
    params['resdir'] = os.path.join(current_dir, 'data', 'tile')
    params['driver_bin'] = "C:/Users/vemun/PycharmProjects/Master/osmaug/geckodriver.exe"
    getosm_function(params)


def test():
    current_dir = os.path.abspath(os.getcwd())
    config_dir = os.path.join(current_dir, 'config')
    f_config = open(os.path.join(config_dir, 'collosm.json'), encoding='utf8')
    params = json.load(f_config)
    params['workdir'] = os.path.join(current_dir, 'data', 'tile')
    params['resdir'] = os.path.join(current_dir, 'data')
    params['file']['name'] = os.path.join(current_dir, 'data', 'tab', 'NO2_oslo10_test.csv')
    collosm_function(params)


def idk():
    current_dir = os.path.abspath(os.getcwd())
    config_dir = os.path.join(current_dir, 'config')
    f_config = open(os.path.join(config_dir, 'testaugmosm.json'), encoding='utf8')
    params = json.load(f_config)
    params['workdir'] = os.path.join(current_dir, 'data', 'tile')
    params['resdir'] = os.path.join(current_dir, 'data', 'img')
    augmosm_function(params)


if __name__ == '__main__':
    main_heavy()


