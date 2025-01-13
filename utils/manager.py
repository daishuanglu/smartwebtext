import os

def incompleted(saved_dir, fpattern, min_doc):
    ret = os.popen("find {:s} -name '{:s}' | xargs wc -l".format(saved_dir, fpattern))
    status_profile = ret.readlines()
    if len(status_profile) == 1 and status_profile[0].strip() == '0':
        return
    incomplete_ids = []
    for count_line in status_profile[:-1]:
        cnt, thread = count_line.strip().split()
        print(cnt, thread)
        cnt = int(cnt)
        thread_id = int(thread.split('_')[-1][:-4])
        if int(cnt) < min_doc - 1:
            incomplete_ids.append(thread_id)
    return incomplete_ids

