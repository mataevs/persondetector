
def generate_flag_file(experiment_path, metadata_path):
    with open(experiment_path) as f:
        content = f.readlines()

    with open(metadata_path, "w") as mf:
        flags = []
        for line in content:
            if line.find("\"type\": \"skeleton\"") != -1:
                created_at_key = "\"created_at\": "
                created_s = line.find(created_at_key) + len(created_at_key)
                created_e = line.find(",", created_s+1)
                created = line[created_s:created_e]

                sensor_id_key = "\"sensor_id\": \""
                sensor_s = line.find(sensor_id_key) + len(sensor_id_key)
                sensor_e = line.find("\"", sensor_s+1)
                sensor = line[sensor_s:sensor_e]

                flags.append((created, sensor))

                mf.write(created + "," + sensor + "\n")

def generate_rgb_files(experiment_path, image_dir):
    with open(experiment_path) as f:
        content = f.readlines()

    for line in content:
        if line.find("\"type\": \"image_rgb\"") != -1:
            created_at_key = "\"created_at\": "
            created_s = line.find(created_at_key) + len(created_at_key)
            created_e = line.find(",", created_s+1)
            created = line[created_s:created_e]

            sensor_id_key = "\"sensor_id\": \""
            sensor_s = line.find(sensor_id_key) + len(sensor_id_key)
            sensor_e = line.find("\"", sensor_s+1)
            sensor = line[sensor_s:sensor_e]

            image_key = "\"image\": \""
            image_s = line.find(image_key) + len(image_key)
            image_e = line.find("\"", image_s + 1)
            image = line[image_s:image_e]

            fimg = open(image_dir + "/" + created + "_" + sensor + ".jpg", "wb")
            fimg.write(image.decode('base64'))
            fimg.close()

# generate_flag_file("/home/mataevs/ptz/ptz_test_1", "metadata.csv")
generate_rgb_files("/home/mataevs/ptz/ptz_test_1", "rgbs")