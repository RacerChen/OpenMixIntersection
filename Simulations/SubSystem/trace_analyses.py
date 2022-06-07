from xml.etree import ElementTree as ET
from EntityClass.VehicleTrace import VehicleTrace
from tlLogic_gen import GS_y_duration

# Global Setting
GS_rou_xml_file = 'cross_cjj.rou.xml'
GS_battery_xml_file = 'batteryConsume.xml'
GS_trace_xml_file = 'sumoTrace.xml'


# Part I
# Step1: rou文件初始单辆车的信息
def analyse_rou(rou_xml):
    # Step1: 在rou文件中读取车辆基本信息
    vehicletrace_dict = {}
    rou_tree = ET.parse(rou_xml)
    rou_root = rou_tree.getroot()
    vehicles = rou_root.findall('vehicle')
    for vehicle in vehicles:
        v_attrib = vehicle.attrib
        v_type = v_attrib['type'].split('_')
        v_route = v_attrib['route'].split('_')
        vehicletrace_dict[v_attrib['id']] = VehicleTrace(v_attrib['id'], v_type[0], v_type[1], v_type[2],
                                                         v_type[3], v_route[0], v_route[1])
    print('init 1 finished...')
    return vehicletrace_dict


# Step2：初始化Vehicle的电能消耗与电能回收量
def analyse_battery_consume(rou_xml, battery_xml):
    vehicletrace_dict = analyse_rou(rou_xml)

    battery_tree = ET.parse(battery_xml)
    battery_root = battery_tree.getroot()

    timesteps = list(battery_root.iter('timestep'))

    for ts in timesteps:
        vehicles = list(ts.iter('vehicle'))
        for vehicle in vehicles:
            v_attrib = vehicle.attrib
            vid = v_attrib['id']
            vehicletrace_dict[vid].init_battery(float(v_attrib['totalEnergyConsumed']),
                                                float(v_attrib['totalEnergyRegenerated']))
            vehicletrace_dict[vid].energyConsumed_list.append(float(v_attrib['energyConsumed']))
            # 每次init最新的数值，最后存储的值即为最终总值
    print('init 2 finished...')
    return vehicletrace_dict


# Step3： 在SUMO生成的FCD trace文件中初始化VehicleTrace类的trace_df，并生成5项性能指标
def analyse_trace(rou_xml, battery_xml, trace_xml):
    """

    :param rou_xml: rou文件路径
    :param battery_xml:  batteryConsume文件路径
    :param trace_xml: trace文件路径
    :return:
    """
    vehicletrace_dict = analyse_battery_consume(rou_xml, battery_xml)
    tree = ET.parse(trace_xml)
    root = tree.getroot()

    timesteps = list(root.iter('timestep'))

    for ts in timesteps:
        timestep = ts.attrib['time']

        vehicles = list(ts.iter('vehicle'))

        for vehicle in vehicles:
            v_attrib = vehicle.attrib
            vid = v_attrib['id']
            vehicletrace_dict[vid].timestep_list.append(float(timestep))
            vehicletrace_dict[vid].x_list.append(float(v_attrib['x']))
            vehicletrace_dict[vid].y_list.append(float(v_attrib['y']))
            vehicletrace_dict[vid].angle_list.append(v_attrib['angle'])
            vehicletrace_dict[vid].speed_list.append(float(v_attrib['speed']))
            vehicletrace_dict[vid].pos_list.append(float(v_attrib['pos']))
            vehicletrace_dict[vid].lane_list.append(v_attrib['lane'])
            vehicletrace_dict[vid].slope_list.append(float(v_attrib['slope']))
            vehicletrace_dict[vid].signals_list.append(v_attrib['signals'])
            vehicletrace_dict[vid].acceleration_list.append(float(v_attrib['acceleration']))

            vehicletrace_dict[vid].leader_ids_list.append(v_attrib['leaderID'])
            vehicletrace_dict[vid].leader_speeds_list.append(float(v_attrib['leaderSpeed']))
            vehicletrace_dict[vid].leader_gaps_list.append(float(v_attrib['leaderGap']))

    for key in vehicletrace_dict.keys():
        vehicletrace_dict[key].init_df()
        # vehicletrace_dict[key].init_intersection_performance()

    print('init 3 finished...')

    return vehicletrace_dict


# Part II
# 根据所有车生成汇总的性能指标
def stat_all_performance_indexes(vehicletrace_dict):
    all_performance_indexes = {
        'avg_crossing_time': 0,
        'avg_stopping_time': 0,
        'avg_crossing_avg_speed': 0,
        'avg_acc_changing_rate': 0,
        'avg_avg_gap': 0,
        'avg_energy_used': 0,
        'avg_energy_regen': 0,
    }

    vehicle_count = 0
    for key in vehicletrace_dict.keys():
        vehicle_count += 1
        all_performance_indexes['avg_crossing_time'] += vehicletrace_dict[key].crossing_time
        all_performance_indexes['avg_stopping_time'] += vehicletrace_dict[key].stopping_time
        all_performance_indexes['avg_crossing_avg_speed'] += vehicletrace_dict[key].crossing_avg_speed
        all_performance_indexes['avg_acc_changing_rate'] += vehicletrace_dict[key].acc_changing_rate
        all_performance_indexes['avg_avg_gap'] += vehicletrace_dict[key].center_avg_gap
        all_performance_indexes['avg_energy_used'] += vehicletrace_dict[key].totalEnergyConsumed
        all_performance_indexes['avg_energy_regen'] += vehicletrace_dict[key].totalEnergyRegenerated

    for key in all_performance_indexes.keys():
        all_performance_indexes[key] /= vehicle_count
        print('%s:%f' % (key, all_performance_indexes[key]))


def trace_df_to_csv(rou_xml, battery_xml, trace_xml, tl_list, composition_dir):
    """
    将带有相位时间段tag的trace df输出保存
    :param rou_xml:
    :param battery_xml:
    :param trace_xml:
    :param tl_list: 信号灯相位时长
    :param composition_dir: 保存文件目录
    :return:
    """
    vehicletrace_dict = analyse_trace(rou_xml, battery_xml, trace_xml)
    for key in vehicletrace_dict.keys():
        vehicletrace_dict[key].trace_df_parsed_tag_gen(tl_list)
        vehicletrace_dict[key].trace_df.to_csv(composition_dir + '/' + key + '.csv', index=False)


if __name__ == '__main__':
    vd = analyse_trace(GS_rou_xml_file, GS_battery_xml_file, GS_trace_xml_file)
    stat_all_performance_indexes(vd)
    vd['0'].trace_df_parsed_tag_gen([10, 20, 10, 20])

