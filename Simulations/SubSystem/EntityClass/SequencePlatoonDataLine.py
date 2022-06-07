from EntityClass.TimestepVehicle import TimestepVehicle
from rou_gen import VP_vehicle_maxSpeed
from TimestepLanePlatoon import GS_West_East, GS_North_South
import numpy as np

# Global Setting
GS_driver_type = {
    'auto': 0,
    'man': 1
}

GS_ahead_vehicle_type = {
    'auto': 0,
    'man': 1,
    'none': 2
}

GS_tag_float_round = 3


class SequencePlatoonDataLine:
    def __init__(self, cur_lane, next_control_dur, next_control_phase, auto_platoon_timestep_vehicles_list,
                 vehicle_cross_info_dict, v_rou_info_dict):
        self.cur_lane = cur_lane
        self.next_control_dur = next_control_dur
        self.next_control_phase = next_control_phase
        self.auto_platoon_timestep_vehicles_list = auto_platoon_timestep_vehicles_list
        self.vehicle_cross_info_dict = vehicle_cross_info_dict
        self.vehicle_rou_info_dict = v_rou_info_dict

    def __str__(self):
        return 'Lane: %s, next dur:%s, next phase:%s' % (self.cur_lane, self.next_control_dur, self.next_control_phase)

    def printVehicles(self):
        for tsv in self.auto_platoon_timestep_vehicles_list:
            print(tsv)

    def getDataLineList(self):
        data_line_list = []

        for tsv in self.auto_platoon_timestep_vehicles_list:
            # FEATURES
            cur_router = int(self.cur_lane[1])  # router number ONE HOT
            cur_lane = int(self.cur_lane[3])  # lane number ONE HOT
            # data_line_list.append(cur_router)
            for i in range(4):  # four means four roads
                if i == cur_router:
                    data_line_list.append(1.0)
                else:
                    data_line_list.append(0.0)

            # data_line_list.append(cur_lane)
            for i in range(4):  # four means max lane count
                if i == cur_lane:
                    data_line_list.append(1.0)
                else:
                    data_line_list.append(0.0)
            data_line_list.append(float(self.next_control_dur))
            # data_line_list.append(int(self.next_control_phase))  # traffic light index ONE HOT
            cur_next_control_phase = int(self.next_control_phase)
            for i in range(8):
                if i == cur_next_control_phase:
                    data_line_list.append(1.0)
                else:
                    data_line_list.append(0.0)
            data_line_list.append(float(tsv.x))
            data_line_list.append(float(tsv.y))
            data_line_list.append(float(tsv.angle))
            data_line_list.append(float(tsv.speed))
            data_line_list.append(float(tsv.pos))
            data_line_list.append(float(tsv.slope))
            # data_line_list.append(tsv.signal)  # int ONE HOT
            for i in range(13):
                if i == int(tsv.signal):
                    data_line_list.append(1.0)
                else:
                    data_line_list.append(0.0)
            data_line_list.append(tsv.acceleration)
            data_line_list.append(float(self.vehicle_rou_info_dict[str(tsv.vid)]['accl']))
            data_line_list.append(float(self.vehicle_rou_info_dict[str(tsv.vid)]['decl']))
            data_line_list.append(float(self.vehicle_rou_info_dict[str(tsv.vid)]['vlen']))
            # data_line_list.append(int(self.vehicle_rou_info_dict[str(tsv.vid)]['start'][1]))  # router number ONE HOT
            cur_start_road = int(self.vehicle_rou_info_dict[str(tsv.vid)]['start'][1])
            for i in range(1, 5):
                if i == cur_start_road:
                    data_line_list.append(1.0)
                else:
                    data_line_list.append(0.0)

            # data_line_list.append(int(self.vehicle_rou_info_dict[str(tsv.vid)]['end'][1]))  # router number ONE HOT
            cur_end_road = int(self.vehicle_rou_info_dict[str(tsv.vid)]['end'][1])
            for i in range(5, 9):
                if i == cur_end_road:
                    data_line_list.append(1.0)
                else:
                    data_line_list.append(0.0)

            if tsv.leaderGap == -1.0:
                # data_line_list.append(GS_ahead_vehicle_type['none'])  # head type ONE HOT:2 前面没车
                cur_ahead_type = GS_ahead_vehicle_type['none']
                for i in range(3):
                    if i == cur_ahead_type:
                        data_line_list.append(1.0)
                    else:
                        data_line_list.append(0.0)
                data_line_list.append(VP_vehicle_maxSpeed)  # leader speed
                if tsv.lane[1] == '1' or tsv.lane[1] == '3':
                    data_line_list.append(float(GS_North_South - tsv.pos))  # intersection gap
                elif tsv.lane[1] == '2' or tsv.lane[1] == '4':
                    data_line_list.append(float(GS_West_East - tsv.pos))  # intersection gap
                else:
                    print('ERROR')
                    return
            else:
                # data_line_list.append(GS_ahead_vehicle_type[self.vehicle_rou_info_dict[str(int(tsv.leaderId))]['driver']])
                cur_ahead_type = GS_ahead_vehicle_type[self.vehicle_rou_info_dict[str(int(tsv.leaderId))]['driver']]
                for i in range(3):
                    if i == cur_ahead_type:
                        data_line_list.append(1.0)
                    else:
                        data_line_list.append(0.0)
                # leader type
                data_line_list.append(float(tsv.leaderSpeed))  # leader speed
                data_line_list.append(float(tsv.leaderGap))  # leader gap

            # TAG
            pass_ts = self.vehicle_cross_info_dict[tsv.vid][0]  # pass timestep
            cur_ts = tsv.timestep
            pass_use_ts = pass_ts - cur_ts
            if pass_use_ts <= float(self.next_control_dur):
                data_line_list.append(round(pass_use_ts, GS_tag_float_round))
            else:
                data_line_list.append(round(float(self.next_control_dur) + 1, GS_tag_float_round))  # not *2 but +1
        # print(data_line_list)
        return data_line_list


if __name__ == '__main__':
    # a =[1,2,3]
    # b=[4,5,6]
    # print(a+b)
    a = 'r1_1'
    print(a[0])
