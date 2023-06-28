from datasets.CILdataset import *


def shapenet2modelnet():
    session_maker = SessionMaker()
    shapenet_train = ShapeNetCIL(partition='train')
    shapenet_test = ShapeNetCIL(partition='test')
    shapenet_id2name = ShapeNetCIL.id2name
    modelnet_banlist = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'guitar', 'lamp', 'laptop', 'piano', 'sofa', 'table', 'keyboard']
    modelnet_train = ModelNet40AlignCIL(partition='train', banlist=modelnet_banlist)
    modelnet_test = ModelNet40AlignCIL(partition='test', banlist=modelnet_banlist)
    modelnet_id2name = ModelNet40AlignCIL.id2name
    session_maker.append_dataset_train_test(shapenet_train, shapenet_test, shapenet_id2name)
    session_maker.append_dataset_train_test(modelnet_train, modelnet_test, modelnet_id2name, modelnet_train.get_load_method())
    session_maker.set_session(num_base_cat=55, num_inc_cat=4)
    return session_maker

def null2modelnet():
    session_maker = SessionMaker()
    modelnet_banlist = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'guitar', 'lamp', 'laptop', 'piano', 'sofa', 'table', 'keyboard']
    modelnet_train = ModelNet40AlignCIL(partition='train', banlist=modelnet_banlist)
    modelnet_test = ModelNet40AlignCIL(partition='test', banlist=modelnet_banlist)
    modelnet_id2name = ModelNet40AlignCIL.id2name
    session_maker.append_dataset_train_test(modelnet_train, modelnet_test, modelnet_id2name, modelnet_train.get_load_method())
    session_maker.set_session(num_base_cat=0, num_inc_cat=4)
    return session_maker

def save_settings(maker):
    for t in range(0, maker.tot_session()):
        train, test = maker.make_session(t, update_memory=0)
        d1 = './data/index_files/shapenet_modelnet/'+str(t)
        d2 = './data/index_files/shapenet_modelnet/'+str(t)
        if not os.path.exists(d1):
            os.makedirs(d1)
        if not os.path.exists(d2):
            os.makedirs(d2)
        train.save(d1+'/train.txt')
        test.save(d2+'/test.txt')

if __name__ == '__main__':
    maker = shapenet2modelnet()
    save_settings(maker)
    for t in range(1, maker.tot_session()):
        train, test = maker.make_session(t, update_memory=0)
        d = './data/index_files/check/'+ str(t)
        print(f't:{t}')
        print('train:')
        train.check(d+'/train.txt')
        print('test:')
        test.check(d+'/test.txt')
        
    
    