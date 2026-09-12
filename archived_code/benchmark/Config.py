# Run from the PAIR repository root; place the released files under data/.
import json
color_map = json.load(open('data/color.json'))


class benchmark_config(): 
    def __init__(self):
        self.watch_dir = 'data/watch_cleaned'
        self.recall_dir = 'data/recall_cleaned'
        self.watch_PSD_DE = 'data/watch_PSD_DE'
        self.recall_PSD_DE = 'data/recall_PSD_DE'
        
        self.GT_label = 'data/GT_label.npy'
        self.optical_flow_score = 'data/optical_flow_score.npy'
        self.video_features = 'data/video_analysis_results.npy'

        self.train_ratio = 0.8
        self.batch_size = 16
        self.num_workers = 4
        self.optical_flow_threshold = 0.6427
        
        self.map_color = color_map
        self.color2num = {"Neutral Light": 0, "Earth & Dark": 1, "Cool Tones": 2, "Green Nature": 3, "Warm Vibrant": 4}


# 20 classes
'''
自然景观 (Natural Landscape):
    动物 (Animals): 1
    植物 (Plants): 2
    水 (Water): 3
    山脉 (Mountains): 4
    天气 (Weather): 5
人类行为 (Human Activities): 
    笑容 (Smiling): 6
    跑步 (Running): 7
    看书 (Reading): 8
    交谈 (Conversation): 9
    吃饭 (Eating): 10
人造物品 (Man-made Objects):
    电子产品 (Electronics): 11
    家具 (Furniture): 12
    交通工具 (Vehicles): 13
    衣物 (Clothing): 14
    娱乐用品 (Recreational Items): 15
复合场景 (Complex Scenes):
    会议 (Meeting): 16
    节日 (Festival): 17
    竞赛 (Competition): 18
    游行 (Parade): 19
    灾难 (Disaster): 20
'''