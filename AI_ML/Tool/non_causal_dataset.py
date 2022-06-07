# Transfer sequential dataset to non-causal ones
import pandas as pd
import csv


def seq2nonseq(vehicle_count, autonomouos_ratio, version):
    print('runing...')
    with open('../seq_dir/' + str(vehicle_count) + '_' + str(autonomouos_ratio)
              + '_Auto_Platoon_Dataset_' + version + '_features.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # print(row)
            temp_f_line = []
            for i in range(len(row)):
                temp_f_line.append(row[i])
                if len(temp_f_line) == 53:
                    f_f = open('../noseq_dir/' + str(vehicle_count) +
                               '_' + str(autonomouos_ratio) + '_Auto_Platoon_Dataset_'
                               + version + '_noseq_features.csv', 'a', newline='')
                    csv_f_writer = csv.writer(f_f)
                    csv_f_writer.writerow(temp_f_line)
                    f_f.close()
                    temp_f_line = []

    with open('../noseq_dir/' + str(vehicle_count) + '_' + str(autonomouos_ratio)
              + '_Auto_Platoon_Dataset_' + version + '_tags.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # print(row)
            for i in range(len(row)):
                f_t = open('../noseq_dir/' + str(vehicle_count) +
                           '_' + str(autonomouos_ratio) + '_Auto_Platoon_Dataset_'
                           + version + '_noseq_tags.csv', 'a', newline='')
                csv_f_writer = csv.writer(f_t)
                csv_f_writer.writerow([row[i]])
                f_t.close()


if __name__ == '__main__':
    seq2nonseq(100, 0.2, 'V5')
    seq2nonseq(100, 0.4, 'V5')
    seq2nonseq(100, 0.6, 'V5')
    seq2nonseq(100, 0.8, 'V5')
    seq2nonseq(100, 1.0, 'V5')
    seq2nonseq(50, 0.5, 'V5')
    seq2nonseq(150, 0.5, 'V5')
    seq2nonseq(200, 0.5, 'V5')
    seq2nonseq(250, 0.5, 'V5')
