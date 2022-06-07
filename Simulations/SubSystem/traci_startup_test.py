from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import sys
import optparse
import random
from Traci import rou_gen

# Global Setting
from trace_analyses import analyse_trace, trace_df_to_csv
from tlLogic_gen import GS_y_duration

GS_output_battery_consume_dir = 'D:/Experiment_Data/battery_consume'
GS_output_sumo_trace_dir = 'D:/Experiment_Data/sumo_trace_files'
GS_output_trace_df_parsed_df = 'D:/Experiment_Data/trace_parsed_df'

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa


def run():
    """execute the TraCI control loop"""
    step = 0

    # we start with phase 2 where EW has green
    traci.trafficlight.setPhase("gneJ8", 2)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        # print('----------------')
        # print(step)
        # next_switch = traci.trafficlight.getNextSwitch("gneJ8")
        # print(next_switch)
        # # phase = traci.trafficlight.getRedYellowGreenState("gneJ8")
        # # print(phase)
        # phase_duration = traci.trafficlight.getPhaseDuration("gneJ8")
        # print(phase_duration)
        # print('----------------')

        # for veh_id in traci.vehicle.getIDList():
        #     print(veh_id)
        #     speed = traci.vehicle.getSpeed(veh_id)
        #     print(speed)
        #     cue_lane = traci.vehicle.getLaneID(veh_id)
        #     print(cue_lane)
        #     print('---------------')
        # if traci.trafficlight.getPhase("gneJ8") == 2:
        #     traci.trafficlight.setPhase("gneJ8", 3)
        # else:
        #     traci.trafficlight.setPhase("gneJ8", 3)
        step += 0.1
    traci.close()
    # sys.stdout.flush()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


def gen_simulating_raw_dataset(rou_xml_dir, add_xml_dir):
    temp_filename = rou_xml_dir.split('/')[1]
    total_vehicle = temp_filename.split('_')[0]
    auto_ratio = temp_filename.split('_')[-1]
    print(total_vehicle)
    print(auto_ratio)

    output_trace_df_parsed_df = GS_output_trace_df_parsed_df + '/' + str(total_vehicle) \
                                + '_vehicle_of_auto_ratio_' + str(auto_ratio)
    if not os.path.exists(output_trace_df_parsed_df):
        os.mkdir(output_trace_df_parsed_df)

    # battery_dir = GS_output_battery_consume_dir + '/' + str(total_vehicle) \
    #               + '_vehicle_of_auto_ratio_' + str(auto_ratio)
    # if os.path.exists(battery_dir):
    #     shutil.rmtree(battery_dir)
    # os.mkdir(battery_dir)
    #
    # trace_dir = GS_output_sumo_trace_dir + '/' + str(total_vehicle) \
    #               + '_vehicle_of_auto_ratio_' + str(auto_ratio)
    # if os.path.exists(trace_dir):
    #     shutil.rmtree(trace_dir)
    # os.mkdir(trace_dir)

    for rou_xml in os.listdir(rou_xml_dir):
        print(rou_xml)

        for add_xml in os.listdir(add_xml_dir):
            add_id = str(add_xml.split('-')[0])
            tls_dur_list = list(add_xml.split('-')[1].split('_'))

            # trace_filename = trace_dir + '/' + str(rou_xml.split('.')[0]) + '_' + add_id + ".sumoTrace.xml"
            # battery_filename = battery_dir + '/' + str(rou_xml.split('.')[0]) + '_' + add_id + ".batteryConsume.xml"
            trace_filename = "tempSumoTrace.xml"
            battery_filename = "tempBatteryConsume.xml"

            traci.start(["D:/软件/SUMO/bin/sumo",
                         "-c", "cross_cjj.sumo.cfg",
                         "--route-files", rou_xml_dir + '/' + rou_xml,
                         "-d", "0",
                         "-a", add_xml_dir + '/' + add_xml,
                         "--fcd-output.signals", "True",  # 生成车辆状态信息，比如变道
                         "--fcd-output.acceleration", "True",  # 生成加减速速度信息
                         "--fcd-output.max-leader-distance", "1",  # 生成前车距离
                         "--fcd-output", trace_filename,  # 生成路径文件
                         "--device.battery.probability", "1",  # 为每一辆车辆装备电池，用于仿真能源消耗
                         "--battery-output", battery_filename,  # 将能源消耗输出到batteryConsume.xml文件中
                         # "--tripinfo-output", "tripinfo.xml"
                         ])
            run()
            cur_output_trace_df_parsed_df = output_trace_df_parsed_df + '/' + str(rou_xml.split('.')[0]) + '_' + add_id
            if not os.path.exists(cur_output_trace_df_parsed_df):
                os.mkdir(cur_output_trace_df_parsed_df)
            trace_df_to_csv(rou_xml_dir + '/' + rou_xml, battery_filename, trace_filename,
                            tls_dur_list, cur_output_trace_df_parsed_df)


# this is the main entry point of this script
if __name__ == "__main__":
    # options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    # if options.nogui:
    #     sumoBinary = checkBinary('sumo')
    # else:
    #     sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    # rou_gen.generate_routefile('cross_cjj.rou.xml')

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    # traci.start(["D:/软件/SUMO/bin/sumo",
    #              "-c", "cross_cjj.sumo.cfg",
    #              "--route-files", "cross_cjj.rou.xml",
    #              "-d", "10",
    #              "-a", "tls.add.xml",
    #              "--fcd-output.signals", "True",  # 生成车辆状态信息，比如变道
    #              "--fcd-output.acceleration", "True",  # 生成加减速速度信息
    #              "--fcd-output.max-leader-distance", "1",  # 生成前车距离
    #              "--fcd-output", "sumoTrace.xml",  # 生成路径文件
    #              "--device.battery.probability", "1",  # 为每一辆车辆装备电池，用于仿真能源消耗
    #              "--battery-output", "batteryConsume.xml",  # 将能源消耗输出到batteryConsume.xml文件中
    #              "--tripinfo-output", "tripinfo.xml"
    #              ])
    # run()

    # gen_simulating_raw_dataset('rou_xml_file/50_vehicle_of_auto_ratio_0.5', 'tls_add_files')
    # gen_simulating_raw_dataset('rou_xml_file/150_vehicle_of_auto_ratio_0.5', 'tls_add_files')
    # gen_simulating_raw_dataset('rou_xml_file/200_vehicle_of_auto_ratio_0.5', 'tls_add_files')
    gen_simulating_raw_dataset('rou_xml_file/250_vehicle_of_auto_ratio_0.5', 'tls_add_files')
    # gen_simulating_raw_dataset('rou_xml_file/300_vehicle_of_auto_ratio_0.5', 'tls_add_files')
