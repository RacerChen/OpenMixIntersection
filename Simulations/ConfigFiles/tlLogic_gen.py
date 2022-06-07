from xml.etree import ElementTree as ET
import itertools
import os
import shutil

# Global Setting
GS_loop = 4  # 一个相位有几个循环
GS_y_duration = 3
GS_phase_codes_dict = {
    # ['行驶相位', '黄灯相位']
    '1': ['GGGrgrrrrGGGrgrrrr', 'gyyrgyyyrgyyrgyyyr'],
    '2': ['grrrGGGGrgrrrGGGGr', 'gyyrgyyyrgyyrgyyyr'],
    '3': ['GrrGGrrrrGrrGGrrrr', 'GrryGrrrrGrryGrrrr'],
    '4': ['GrrrGrrrGGrrrGrrrG', 'GrrrGrrryGrrrGrrry'],

    '5': ['GGGGGrrrrGrrrGrrrr', 'gyyygrrrrgrrrgrrrr'],
    '6': ['GrrrGGGGGGrrrGrrrr', 'grrrgyyyygrrrgrrrr'],
    '7': ['GrrrGrrrrGGGGGrrrr', 'grrrgrrrrgyyygrrrr'],
    '8': ['GrrrGrrrrGrrrGGGGG', 'grrrgrrrrgrrrgyyyy']
}
GS_tls_add_dir = 'D:/Experiment_Data/tls_add_files'

GS_min_dur = 10
GS_max_dur = 30
GS_step_dur = 5


def durations_series_gen(min_dur, max_dur, step_dur):
    """
    生成min dur到max dur的时长，步长为step dur
    :param min_dur: 最小时长
    :param max_dur: 最大时长（不包含）
    :param step_dur: 步长
    :return:
    """
    durations = []
    for i in range(min_dur, max_dur, step_dur):
        for j in range(min_dur, max_dur, step_dur):
            temp_duration = [i, max_dur - i, j, max_dur - j]
            durations.append(temp_duration)
    return durations


def phase_series_gen():
    """
    生成1-4，5-8的相位排列序列
    :return:
    """
    phases_1_4 = []
    phase_indexes_1_4 = []
    for item_1_4 in itertools.permutations('1234', 4):
        phase_indexes_1_4.append('%s_%s_%s_%s' % (item_1_4[0], item_1_4[1], item_1_4[2], item_1_4[3]))
        phases_1_4.append([GS_phase_codes_dict[item_1_4[0]], GS_phase_codes_dict[item_1_4[1]],
                           GS_phase_codes_dict[item_1_4[2]], GS_phase_codes_dict[item_1_4[3]]])
    phases_5_8 = []
    phase_indexes_5_8 = []
    for item_5_8 in itertools.permutations('5678', 4):
        phase_indexes_5_8.append('%s_%s_%s_%s' % (item_5_8[0], item_5_8[1], item_5_8[2], item_5_8[3]))
        phases_5_8.append([GS_phase_codes_dict[item_5_8[0]], GS_phase_codes_dict[item_5_8[1]],
                           GS_phase_codes_dict[item_5_8[2]], GS_phase_codes_dict[item_5_8[3]]])
    return phases_1_4, phases_5_8, phase_indexes_1_4, phase_indexes_5_8


def tlLogic_gen(duration_list, phase_list):
    add_xml_str = '<additional>\n'
    add_xml_str += '\t<tlLogic id=\"gneJ8\" type=\"static\" programID=\"1\" offset=\"0\">\n'
    for i in range(0, GS_loop):
        add_xml_str += '\t\t<phase duration="%d" state="%s"/>\n' % (duration_list[i], phase_list[i][0])
        add_xml_str += '\t\t<phase duration="%d" state="%s"/>\n' % (GS_y_duration, phase_list[i][1])  # 黄灯相位
    add_xml_str += '\t</tlLogic>\n'
    add_xml_str += '</additional>\n'
    return add_xml_str


def gen_tl_add_xmls():
    if os.path.exists(GS_tls_add_dir):
        shutil.rmtree(GS_tls_add_dir)
    os.mkdir(GS_tls_add_dir)
    durs = durations_series_gen(GS_min_dur, GS_max_dur, GS_step_dur)
    phs_1_4, phs_5_8, ph_i_1_4, ph_i_5_8 = phase_series_gen()
    print(durs)
    print(phs_1_4)
    print(phs_5_8)
    file_index = 0
    for dur in durs:
        dur_name = '%d_%d_%d_%d' % (dur[0], dur[1], dur[2], dur[3])
        for ph_1_4_i in range(0, len(phs_1_4)):
            cur_add_xml = tlLogic_gen(dur, phs_1_4[ph_1_4_i])
            add_xml = open('%s/%d-%s-%s.add.xml' % (GS_tls_add_dir, file_index, dur_name, ph_i_1_4[ph_1_4_i]), 'w')
            add_xml.write(cur_add_xml)
            add_xml.close()
            file_index += 1
        for ph_5_8_i in range(0, len(phs_5_8)):
            cur_add_xml = tlLogic_gen(dur, phs_5_8[ph_5_8_i])
            add_xml = open('%s/%d-%s-%s.add.xml' % (GS_tls_add_dir, file_index, dur_name, ph_i_5_8[ph_5_8_i]), 'w')
            add_xml.write(cur_add_xml)
            add_xml.close()
            file_index += 1
    print('add xml gen finished...')


if __name__ == '__main__':
    gen_tl_add_xmls()
