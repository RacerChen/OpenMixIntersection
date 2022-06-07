import pandas as pd


# Global Setting
from tlLogic_gen import GS_y_duration

GS_cross_x_boundary = 60
GS_cross_y_boundary = 80
GS_min_timestep = 0.1
tmp_df_file = 'tmp_df_file.csv'


class VehicleTrace:
    def __init__(self, vid, vdrive, vacl, vdcl, vlen, vstart, vend):
        self.vid = vid
        self.vdrive = vdrive
        self.vacl = float(vacl)
        self.vdcl = float(vdcl)
        self.vlen = float(vlen)
        self.vstart = vstart
        self.vend = vend

        self.totalEnergyConsumed = -1
        self.totalEnergyRegenerated = -1

        self.trace_df = pd.DataFrame(columns=['timestep', 'x', 'y', 'angle', 'speed', 'pos',
                                              'lane', 'slope', 'signals', 'acceleration',
                                              'leaderID', 'leaderSpeed', 'leaderGap',
                                              'energyConsumed', 'tl_loop'])

        self.timestep_list = []
        self.x_list = []
        self.y_list = []
        self.angle_list = []
        self.speed_list = []
        self.pos_list = []  # pos: The running position of the vehicle measured from the start of the current lane.
        self.lane_list = []
        self.slope_list = []
        self.signals_list = []
        self.acceleration_list = []

        self.energyConsumed_list = []

        self.leader_ids_list = []
        self.leader_speeds_list = []
        self.leader_gaps_list = []

        self.crossing_time = -1
        self.stopping_time = -1
        self.crossing_avg_speed = -1
        self.acc_changing_rate = -1
        self.center_avg_gap = -1

    def __str__(self):
        return 'VehicleTrace: {vid:%s, drive:%s, acl:%f, dcl:%f, len:%f start:%s-end:%s, energy: %f(used),%f(regen)}' \
               % (self.vid, self.vdrive, self.vacl, self.vdcl, self.vlen, self.vstart, self.vend,
                  self.totalEnergyConsumed, self.totalEnergyRegenerated)

    def init_battery(self, totalEnergyConsumed, totalEnergyRegenerated):
        self.totalEnergyConsumed = totalEnergyConsumed
        self.totalEnergyRegenerated = totalEnergyRegenerated

    def init_df(self):
        """
        init trace_df
        :return:
        """
        self.trace_df['timestep'] = self.timestep_list
        self.trace_df['x'] = self.x_list
        self.trace_df['y'] = self.y_list
        self.trace_df['angle'] = self.angle_list
        self.trace_df['speed'] = self.speed_list
        self.trace_df['pos'] = self.pos_list
        self.trace_df['lane'] = self.lane_list
        self.trace_df['slope'] = self.slope_list
        self.trace_df['signals'] = self.signals_list
        self.trace_df['acceleration'] = self.acceleration_list

        self.trace_df['energyConsumed'] = self.energyConsumed_list

        self.trace_df['leaderID'] = self.leader_ids_list
        self.trace_df['leaderSpeed'] = self.leader_speeds_list
        self.trace_df['leaderGap'] = self.leader_gaps_list

        # Judge whether crossed the intersection or are crossing
        self.trace_df['incross_x'] = abs(self.trace_df['x']) <= GS_cross_x_boundary
        self.trace_df['incross_y'] = abs(self.trace_df['y']) <= GS_cross_y_boundary
        self.trace_df['crossed'] = self.trace_df['lane'].str.contains(self.vend)
        self.trace_df['incross'] = self.trace_df['incross_x'] & self.trace_df['incross_y'] & ~self.trace_df['crossed']
        del self.trace_df['incross_x']
        del self.trace_df['incross_y']

        # print(self.trace_df)
        # self.trace_df.to_csv('VehicleTraceDF/' + self.vid + '.csv', index=False)

    def init_intersection_performance(self):
        """
        :return: generate 5 performance indexes
        """
        sub_df_incross = self.trace_df[self.trace_df['incross']]
        # 通过路口期间的轨迹dataframe

        self.crossing_time = sub_df_incross['timestep'].iloc[-1] - sub_df_incross['timestep'].iloc[0]
        # FEATURE1: 车辆通过路口时间

        sub_df_stopping = sub_df_incross[(sub_df_incross['speed'] == 0.00) & (sub_df_incross['acceleration'] == 0.00)]

        self.stopping_time = len(sub_df_stopping) * GS_min_timestep
        # FEATURE2: 车辆在路口等待的时间

        self.crossing_avg_speed = sub_df_incross['speed'].sum() / len(sub_df_incross)
        # FEATURE3: 通过路口的平均速度

        acc_incross_list = sub_df_incross['acceleration'].tolist()
        acc_changing_times = 0
        for i in range(1, len(acc_incross_list)):
            if acc_incross_list[i] * acc_incross_list[i-1] > 0:
                acc_changing_times += 1
        self.acc_changing_rate = acc_changing_times / len(sub_df_incross)
        # FEATURE4: 通过路口的加减速变动率

        sub_df_incenter = sub_df_incross[sub_df_incross['lane'].str.contains('center')]
        self.center_avg_gap = max(sub_df_incenter['leaderGap'].sum() / len(sub_df_incenter['leaderGap']), 0)
        # FEATURE5: 通过交叉路口时的平均车距

    def trace_df_parsed_tag_gen(self, tl_list):
        """
        生成在哪个相位时间段的tag
        :param tl_list:
        :return:
        """
        tl_loop_index = 0
        i = 0
        in_y = False
        s_ts = 0
        e_ts = 0
        while i < len(tl_list):
            # print(tl_list)
            if in_y:  # 在黄灯相位
                e_ts += GS_y_duration
            else:  # 不在黄灯相位
                e_ts += float(tl_list[i])
                i += 1
            in_y = not in_y
            # print('----------')
            # print('s_ts:%f' % s_ts)
            # print('e_ts:%f' % e_ts)
            self.trace_df['tl_loop'].loc[(self.trace_df['timestep'] >= s_ts) &
                                         (self.trace_df['timestep'] < e_ts)] = tl_loop_index
            tl_loop_index += 1
            s_ts = e_ts
            if i == len(tl_list) and s_ts <= self.trace_df['timestep'].iloc[-1]:
                i = 0
        # self.trace_df.to_csv(tmp_df_file, index=False)
