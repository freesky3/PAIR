# Run from the PAIR repository root; place the released files under data/.
class benchmark_config(): 
    def __init__(self):
        self.watch_dir = 'data/watch_cleaned'
        self.recall_dir = 'data/recall_cleaned'
        self.watch_PSD_DE = 'data/watch_PSD_DE'
        self.recall_PSD_DE = 'data/recall_PSD_DE'


        self.train_ratio = 0.8
        self.batch_size = 128
        self.num_workers = 4