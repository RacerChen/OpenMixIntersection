import random
import collections
import os
import shutil

# Vehicles Parameters:
VP_vehicle_accelerations = [9.26, 5.56, 3.96, 3.09, 2.53, 2.14, 1.85]  # Calculated from 0-100m/s (3,5,7,9,11,13,15s)
# VP_vehicle_decelerations = [3, 3.4, 3.8, 4.2, 4.6]  # This is comfortable deceleration rate, max can reach 7.5-8m/s^2
VP_vehicle_decelerations = [4.5, 4.9, 5.3, 4.7, 5.9]  # This is comfortable deceleration rate, max can reach 7.5-8m/s^2
VP_vehicle_lengths = [3.8, 4.2, 4.6, 5.0, 5.4]  # Normal vehicles' lengths, 5.4m correspond to van
VP_vehicle_maxSpeed = 22.22  # 80km/h

SUMO_total_time = 90

# len(VP_vehicle_xxx) must== len(SUMO_car_ratio_dict['xxx_ratio'])
SUMO_car_ratio_dict = {
    'accel_ratio': [1, 3, 5, 8, 4, 2, 1],  # skewed distribution for electric vehicles accelerate fast.
    'decel_ratio': [1, 2, 4, 2, 1],
    'vlen_ratio': [1, 2, 4, 2, 1]
}

SUMO_car_following_model_dict = {
    'man': 'Krauss',
    'auto': 'CACC'
}

SUMO_car_color_dict = {
    'man': '0,0,1',  # blue
    'auto': '1,1,1'  # white
}

SUMO_auto_ratio = 0.5  # Ratio of automated vehicle among the whole system.
SUMO_driving_ratio_dict = {
    'man': 1 - SUMO_auto_ratio,
    'auto': SUMO_auto_ratio
}

SUMO_total_vehicle_count = 300

SUMO_edge_list = ['r1_r6', 'r1_r5', 'r1_r7',
                  'r2_r7', 'r2_r6', 'r2_r8',
                  'r3_r8', 'r3_r7', 'r3_r5',
                  'r4_r5', 'r4_r8', 'r4_r6']

SUMO_full_edge_dict = {
    'r1_r6': 'r1 r6',  # straight
    'r1_r5': 'r1 r5',  # turn right
    'r1_r7': 'r1 r7',  # turn left

    'r2_r7': 'r2 r7',  # straight
    'r2_r6': 'r2 r6',  # turn right
    'r2_r8': 'r2 r8',  # turn left

    'r3_r8': 'r3 r8',  # straight
    'r3_r7': 'r3 r7',  # turn right
    'r3_r5': 'r3 r5',  # turn left

    'r4_r5': 'r4 r5',  # straight
    'r4_r8': 'r4 r8',  # turn right
    'r4_r6': 'r4 r6',  # turn left
}

SUMO_full_edge_ratio_dict = {
    'r1_r6': 8,  # straight
    'r1_r5': 2,  # turn right
    'r1_r7': 1,  # turn left

    'r2_r7': 5,  # straight
    'r2_r6': 3,  # turn right
    'r2_r8': 2,  # turn left

    'r3_r8': 7,  # straight
    'r3_r7': 1,  # turn right
    'r3_r5': 2,  # turn left

    'r4_r5': 4,  # straight
    'r4_r8': 2,  # turn right
    'r4_r6': 1,  # turn left
}

Class_vType = collections.namedtuple('Class_vType', ['id_str', 'accel', 'decel', 'length', 'car_follow_model',
                                                     'max_speed', 'color', 'probability'])
Class_rType = collections.namedtuple('Class_rType', ['id_str', 'edges', 'probability'])

GS_rou_files_dir = 'rou_xml_file'
GS_rou_file_count = 1


def random_edge_ratio():

    SUMO_full_edge_ratio_dict['r1_r6'] = random.randint(2, 8)
    SUMO_full_edge_ratio_dict['r1_r5'] = random.randint(1, 3)
    SUMO_full_edge_ratio_dict['r1_r7'] = random.randint(1, 3)

    SUMO_full_edge_ratio_dict['r2_r7'] = random.randint(2, 8)
    SUMO_full_edge_ratio_dict['r2_r6'] = random.randint(1, 3)
    SUMO_full_edge_ratio_dict['r2_r8'] = random.randint(1, 3)

    SUMO_full_edge_ratio_dict['r3_r8'] = random.randint(2, 8)
    SUMO_full_edge_ratio_dict['r3_r7'] = random.randint(1, 3)
    SUMO_full_edge_ratio_dict['r3_r5'] = random.randint(1, 3)

    SUMO_full_edge_ratio_dict['r4_r5'] = random.randint(2, 8)
    SUMO_full_edge_ratio_dict['r4_r8'] = random.randint(1, 3)
    SUMO_full_edge_ratio_dict['r4_r6'] = random.randint(1, 3)


# 根据概率输出值
def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            return item


# 生成车辆种类列表
def vType_list_gen():
    vType_list = []

    accel_ratio_list = SUMO_car_ratio_dict['accel_ratio']
    decel_ratio_list = SUMO_car_ratio_dict['decel_ratio']
    length_ratio_list = SUMO_car_ratio_dict['vlen_ratio']

    total_accel_ratio = sum(accel_ratio_list)
    total_decel_ratio = sum(decel_ratio_list)
    total_length_ratio = sum(length_ratio_list)

    for drive_type in SUMO_car_following_model_dict.keys():
        for cur_accel_i in range(0, len(VP_vehicle_accelerations)):
            for cur_decel_i in range(0, len(VP_vehicle_decelerations)):
                for cur_vlen_i in range(0, len(VP_vehicle_lengths)):
                    cur_accel = VP_vehicle_accelerations[cur_accel_i]
                    cur_decel = VP_vehicle_decelerations[cur_decel_i]
                    cur_vlen = VP_vehicle_lengths[cur_vlen_i]
                    cur_probability = (accel_ratio_list[cur_accel_i] / total_accel_ratio) * \
                                      (decel_ratio_list[cur_decel_i] / total_decel_ratio) * \
                                      (length_ratio_list[cur_vlen_i] / total_length_ratio) * \
                                      SUMO_driving_ratio_dict[drive_type]

                    id_str = drive_type + '_' + str(cur_accel) + '_' + str(cur_decel) + '_' + str(cur_vlen)
                    vType_list.append(Class_vType(id_str, cur_accel, cur_decel, cur_vlen,
                                                  SUMO_car_following_model_dict[drive_type], VP_vehicle_maxSpeed,
                                                  SUMO_car_color_dict[drive_type],
                                                  cur_probability))
    print(vType_list)
    return vType_list


# 生成道路种类列表
def rType_list_gen():
    random_edge_ratio()
    rType_list = []
    total_router_ratio = sum(SUMO_full_edge_ratio_dict.values())
    for item in SUMO_edge_list:
        rType_list.append(Class_rType(item, SUMO_full_edge_dict[item],
                                      SUMO_full_edge_ratio_dict[item] / total_router_ratio))
    print(rType_list)
    return rType_list


def generate_routefile(rou_filename):
    vTypes = vType_list_gen()
    rTypes = rType_list_gen()

    with open(rou_filename, 'w') as f:
        vTypes_indexes = [i for i in range(0, len(vTypes))]
        rTypes_indexes = [i for i in range(0, len(rTypes))]
        vTypes_probability = []
        rTypes_probability = []

        print('<routers>', file=f)
        for vType in vTypes:
            vTypes_probability.append(vType.probability)
            print('\t<vType id=\"%s\" accel=\"%f\" decel=\"%f\" length=\"%f\"' \
                  ' carFollowModel=\"%s\" maxSpeed=\"%f\" color=\"%s\" />' % \
                  (vType.id_str, vType.accel, vType.decel, vType.length,
                   vType.car_follow_model, vType.max_speed, vType.color), file=f)

        for rType in rTypes:
            rTypes_probability.append(rType.probability)
            print('\t<route id=\"%s\" edges=\"%s\" />' % (rType.id_str, rType.edges), file=f)

        depart_time_list = []
        for i in range(0, SUMO_total_vehicle_count):
            depart_time_list.append(random.randint(0, SUMO_total_time))
        depart_time_list = sorted(depart_time_list)

        depart_i = 0
        for i in range(0, SUMO_total_vehicle_count):
            vType = vTypes[random_pick(vTypes_indexes, vTypes_probability)]
            rType = rTypes[random_pick(rTypes_indexes, rTypes_probability)]
            print('\t\t<vehicle id=\"%i\" type=\"%s\" route=\"%s\" depart=\"%s\" />' %
                  (depart_i, vType.id_str, rType.id_str, depart_time_list[depart_i]), file=f)
            depart_i += 1

        print('</routers>', file=f)


def gen_all_routefiles(roufile_cnt):
    file_dir = GS_rou_files_dir + '/' + str(SUMO_total_vehicle_count) + '_vehicle_of_auto_ratio_' + str(SUMO_auto_ratio)
    if os.path.exists(file_dir):
        shutil.rmtree(file_dir)
    os.mkdir(file_dir)
    for i in range(0, roufile_cnt):
        random.seed(i)
        generate_routefile(file_dir + '/' + str(i) + '.rou.xml')
        print(str(i) + ' rou file finished...')
    print('All rou files finished...')


if __name__ == '__main__':
    gen_all_routefiles(GS_rou_file_count)
