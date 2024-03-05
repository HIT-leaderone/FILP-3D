from models.base.fscil_trainer import FSCILTrainer as Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utilsx import *
from dataloader.data_utils import *
from .Network import MYNET
from session_settings import shapenet2modelnet,shapenet2modelnet_joint
from datasets.CILdataset import *
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
from torchmetrics.aggregation import MeanMetric
from sklearn.metrics import recall_score,classification_report
from models import FewShotCIL, focal_loss,FewShotCILwoRn2,MYCIL

session_maker = shapenet2modelnet()
id2name = session_maker.get_id2name()

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)
        self.set_up_model(args)
        pass

    def set_up_model(self,args):
        self.model = MYNET(self.args, mode=self.args.base_mode).cuda()
        # print(MYNET)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))

        # if self.args.model_dir != None:  #
        #     print('Loading init parameters from: %s' % self.args.model_dir)
        #     self.best_model_dict = torch.load(self.args.model_dir)['params']
        # else:
        #     print('*********WARNINGl: NO INIT MODEL**********')
        #     # raise ValueError('You must initialize a pre-trained model')
        #     pass

    def update_param(self, model, pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def get_dataloader(self, session, args):
        trainset, testset = session_maker.make_session(session_id=session, update_memory=args.memory_shot)
        print("train_loader=",args.batch_size_new)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_new, num_workers=args.workers,
                                                        pin_memory=args.pin_memory,shuffle=True,drop_last=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_new, num_workers=args.workers,
                                                    pin_memory=args.pin_memory,shuffle=True,drop_last=True)
        
        return trainset, trainloader, testloader

    def get_session_classes(self, session):
        class_list = np.arange(self.args.base_class + session * self.args.way)
        return class_list

    def replace_to_rotate(self, proto_tmp, query_tmp):
        # for i in range(self.args.low_way):
        #     # random choose rotate degree
        #     rot_list = [90, 180, 270]
        #     sel_rot = random.choice(rot_list)
        #     if sel_rot == 90:  # rotate 90 degree
        #         # print('rotate 90 degree')
        #         proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].transpose(2, 3).flip(2)
        #         query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].transpose(2, 3).flip(2)
        #     elif sel_rot == 180:  # rotate 180 degree
        #         # print('rotate 180 degree')
        #         proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].flip(2).flip(3)
        #         query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].flip(2).flip(3)
        #     elif sel_rot == 270:  # rotate 270 degree
        #         # print('rotate 270 degree')
        #         proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].transpose(2, 3).flip(3)
        #         query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].transpose(2, 3).flip(3)
        return proto_tmp, query_tmp

    def get_optimizer_base(self):

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

        return optimizer

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session,args)

            # self.model = self.update_param(self.model, self.best_model_dict)

            if session == 0:  # load base class train img label

                # print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer = self.get_optimizer_base()

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    self.model.eval()
                    tl, ta = self.base_train(self.model, trainloader, optimizer,  epoch, args)

                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)

                # always replace fc with avg mean
                self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)



            else:  # incremental learning sessions
                print("training session: [%d]" % session)
                self.model.load_state_dict(self.best_model_dict)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                tsl, tsa = self.test(self.model, testloader, args, session)

    def validation(self):
        with torch.no_grad():
            model = self.model

            for session in range(1, self.args.sessions):
                train_set, trainloader, testloader = self.get_dataloader(session)

                trainloader.dataset.transform = testloader.dataset.transform
                model.module.mode = 'avg_cos'
                model.eval()
                model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                vl, va = self.test(model, testloader, self.args, session)

        return vl, va

    def base_train(self, model, trainloader, optimizer,  epoch, args):
        tl = Averager()
        ta = Averager()

        tqdm_gen = tqdm(trainloader)

        label = torch.arange(args.episode_way + args.low_way).repeat(args.episode_query)
        label = label.type(torch.cuda.LongTensor)

        for data, label in tqdm(tqdm_gen):
            # label = label.unsqueeze(1).to(device)
            label = label.cuda()
            data = data.cuda()
            # print(data.shape)
            k = int(len(label)/2)
            proto, query = data[:k], data[k:]
            # sample low_way data
            proto_tmp = deepcopy(proto)
            query_tmp = deepcopy(query)
            # random choose rotate degree
            proto_tmp, query_tmp = self.replace_to_rotate(proto_tmp, query_tmp)
            print("data=",data.shape,proto_tmp.shape, query_tmp.shape)
            model.module.mode = 'encoder'
            data = model(data)
            proto_tmp = model(proto_tmp)
            query_tmp = model(query_tmp)
            
            print("data=",data.shape)
            # k = args.episode_way * args.episode_shot
            proto, query = data[:k], data[k:]

            proto = torch.cat([proto, proto_tmp], dim=0)
            proto = proto.mean(0).repeat(16,1)
            query = torch.cat([query, query_tmp], dim=0)

            print(proto.shape,query.shape)

            logits = model.module._forward(proto, query)
            print(logits.shape)
            total_loss = F.cross_entropy(logits, label)

            acc = count_acc(logits, label)

            # lrc = scheduler.get_last_lr()[0]
            # tqdm_gen.set_description(
            #     'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        tl = tl.item()
        ta = ta.item()
        return tl, ta

    def test(self, model, dataloader, args, session):
        test_class = args.base_class + session * args.way
        num_batches = len(dataloader)
        num_cls = test_class
        # define metrics
        metrics = MetricCollection([
            MulticlassAccuracy(num_classes=num_cls, average="micro"),
            MulticlassPrecision(num_classes=num_cls, average="macro"),
            MulticlassRecall(num_classes=num_cls, average="macro")
        ]).cuda()
        confmat = MulticlassConfusionMatrix(num_classes=test_class, normalize='true').cuda()
        
        model.eval()
        predict_list=[]
        ans_list=[]
        novel_tot={}
        novel_acc={}
        if session == 0: last = 0
        else: last = test_class - args.way
        for i in range(last,test_class):
            # print("novel_class=",i)
            novel_tot[i] = 0
            novel_acc[i] = 0
        with torch.no_grad():
            for points, label in tqdm(dataloader):
                # label = label.unsqueeze(1).to(device)
                label = label.cuda()
                points = points.cuda()
                
                query = model(points)
                query = query.unsqueeze(0).unsqueeze(0)

                proto = model.module.fc.weight[:test_class, :].detach()
                proto = proto.unsqueeze(0).unsqueeze(0)

                pred = model.module._forward(proto, query)
                
                # update metrics
                
                confmat.update(pred, label)
                metrics.update(pred, label)
                # label=torch.squeeze(label)
                predict_list.extend(pred.detach().cpu())
                ans_list.extend(label.detach().cpu())
                for i in range(len(pred)):
                    if label[i]>=last:
                        novel_tot[label[i].item()] +=1
                        if pred[i] == label[i]:
                            novel_acc[label[i].item()] +=1
        # print metrics
        acc = metrics.compute()
        print(f"[Test] Task:{session}\t\
                    Accuracy:\t{(100*acc['MulticlassAccuracy']):>0.1f}\t\
                    Precision:\t{(100*acc['MulticlassPrecision']):>0.1f}\t\
                    Recall:\t{(100*acc['MulticlassRecall']):>0.1f}")
        _fig, _ax = confmat.plot(add_text=False)
        # plt.savefig('exp_results/' + exp_time + ':' + args.exp_name + '/task' + str(stat['task_id'])+'_epoch'+str(stat['epoch']) + '.png')
        result = classification_report(ans_list, predict_list, target_names=id2name[:test_class])             
        # io.cprint(result)
        macro_avg = 0
        micro_avg = 0
        tot_novel = 0
        for i in range(last,test_class):
            macro_avg += novel_acc[i]/novel_tot[i]
            micro_avg += novel_acc[i]
            tot_novel += novel_tot[i]
        macro_avg /= (test_class - last)
        micro_avg /= tot_novel
        print(f"[Test] Task:{session}\t\
                    Novel Macro Accuracy:\t{(100*macro_avg):>0.1f}\t\
                    Novel Micro Precision:\t{(100*micro_avg):>0.1f}")
