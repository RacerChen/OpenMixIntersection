import os
from Traci.dynamic_platoon_gen_feature import load_rou_info


class RouInfo:

    def __init__(self, rou_id, SUMO_total_vehicle_count, SUMO_auto_ratio):
        self.rou_info_dict = load_rou_info(SUMO_total_vehicle_count, SUMO_auto_ratio)
        self.rou_id = rou_id
        self.cur_rou_info_dict = load_rou_info(SUMO_total_vehicle_count, SUMO_auto_ratio)[rou_id]
        # dict包含信息：driver, acel, dcel, vlen, start, end
        self.auto_vids = []
        for vid in self.rou_info_dict[rou_id].keys():
            if self.rou_info_dict[rou_id][vid]['driver'] == 'auto':
                self.auto_vids.append(int(vid))

    def getVlen(self, vid):
        return self.cur_rou_info_dict[str(vid)]['vlen']

    def getDriver(self, vid):
        return self.cur_rou_info_dict[str(vid)]['driver']

    def getIsInFinalUseLane(self, vid, cur_lane):
        start = self.cur_rou_info_dict[str(vid)]['start']
        end = self.cur_rou_info_dict[str(vid)]['end']
        cur_lane_id = cur_lane.split('_')[1]
        se = start+end
        if se in ['r1r5', 'r2r6', 'r3r7', 'r4r8']:  # turn right
            if cur_lane_id == '0':
                return True
        elif se in ['r1r7', 'r3r5']:  # turn left.
            if cur_lane_id == '3':
                return True
        elif se in ['r2r8', 'r4r6']:  # turn left.
            if cur_lane_id == '2':
                return True
        elif se in ['r3r8', 'r1r6']:  # straight.
            if cur_lane_id == '1' or cur_lane_id == '2':
                return True
        elif se in ['r2r7', 'r4r5']:  # straight.
            if cur_lane_id == '1':
                return True
        else:
            return False


if __name__ == '__main__':
    GS_North_South_Lane = ['r1', 'r6', 'r3', 'r8']
    if 'r1' in GS_North_South_Lane:
        print('in')
