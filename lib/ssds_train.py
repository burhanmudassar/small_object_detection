from __future__ import print_function
import numpy as np
import os
import sys
import cv2
import random
import pickle

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
import torch.nn.init as init

from tensorboardX import SummaryWriter

from lib.layers import *
from lib.utils.timer import Timer
from lib.utils.data_augment import preproc
from lib.modeling.model_builder import create_model
from lib.dataset.dataset_factory import load_data
from lib.utils.config_parse import cfg
from lib.utils.eval_utils import *
from lib.utils.visualize_utils import *
import matplotlib.pyplot as plt
import glob

class Solver(object):
    """
    A wrapper class for the training process
    """
    def __init__(self, args):
        self.cfg = cfg

        # Load data
        print('===> Loading data')
        self.train_loader = load_data(cfg.DATASET, 'train') if 'train' in cfg.PHASE else None
        self.eval_loader = load_data(cfg.DATASET, 'eval') if 'eval' in cfg.PHASE else None
        self.test_loader = load_data(cfg.DATASET, 'test') if 'test' in cfg.PHASE else None
        # self.test_loader = load_data(cfg.DATASET, 'test') if 'custom_visualize' in cfg.PHASE else None
        self.visualize_loader = load_data(cfg.DATASET, 'visualize') if 'visualize' in cfg.PHASE else None

        # Build model
        print('===> Building model')
        self.model, self.priorbox = create_model(cfg.MODEL)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.detector = Detect(cfg.POST_PROCESS, self.priors)

        # Utilize GPUs for computation
        self.use_gpu = torch.cuda.is_available()
        self.multi_gpu = args.multi_gpu


        if self.use_gpu:
            print('Utilize GPUs for computation')
            print('Number of GPU available', torch.cuda.device_count())
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.cuda()
            self.priors.cuda()
            cudnn.benchmark = True
            if torch.cuda.device_count() > 1 and args.multi_gpu:
                self.model = torch.nn.DataParallel(self.model)
                self.model.to(device)

        # Print the model architecture and parameters
        print('Model architectures:\n{}\n'.format(self.model))

        # print('Parameters and size:')
        # for name, param in self.model.named_parameters():
        #     print('{}: {}'.format(name, list(param.size())))

        # print trainable scope
        print('Trainable scope: {}'.format(cfg.TRAIN.TRAINABLE_SCOPE))
        trainable_param = self.trainable_param(cfg.TRAIN.TRAINABLE_SCOPE)
        self.optimizer = self.configure_optimizer(trainable_param, cfg.TRAIN.OPTIMIZER)
        self.exp_lr_scheduler = self.configure_lr_scheduler(self.optimizer, cfg.TRAIN.LR_SCHEDULER)
        self.max_epochs = cfg.TRAIN.MAX_EPOCHS

        # metric
        self.criterion = MultiBoxLoss(cfg.MATCHER, self.priors, self.use_gpu)

        # Set the logger
        self.writer = SummaryWriter(log_dir=cfg.LOG_DIR)
        self.output_dir = cfg.EXP_DIR
        self.checkpoint = cfg.RESUME_CHECKPOINT
        self.checkpoint_prefix = cfg.CHECKPOINTS_PREFIX


    def save_checkpoints(self, epochs, iters=None):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if iters:
            filename = self.checkpoint_prefix + '_epoch_{:d}_iter_{:d}'.format(epochs, iters) + '.pth'
        else:
            filename = self.checkpoint_prefix + '_epoch_{:d}'.format(epochs) + '.pth'
        filename = os.path.join(self.output_dir, filename)
        if torch.cuda.device_count() > 1 and self.multi_gpu:
            torch.save(self.model.module.state_dict(), filename)
        else:
            torch.save(self.model.state_dict(), filename)
        with open(os.path.join(self.output_dir, 'checkpoint_list.txt'), 'a') as f:
            f.write('epoch {epoch:d}: {filename}\n'.format(epoch=epochs, filename=filename))
        print('Wrote snapshot to: {:s}'.format(filename))

        # TODO: write relative cfg under the same page

    def resume_checkpoint_initial(self, resume_checkpoint):
        if resume_checkpoint == '' or not os.path.isfile(resume_checkpoint):
            print(("=> no checkpoint found at '{}'".format(resume_checkpoint)))
            return False
        print(("=> loading checkpoint '{:s}'".format(resume_checkpoint)))
        checkpoint = torch.load(resume_checkpoint)

        # print("=> Weigths in the checkpoints:")
        # print([k for k, v in list(checkpoint.items())])

        # remove the module in the parallel model
        if 'module.' in list(checkpoint.items())[0][0]:
            pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
            checkpoint = pretrained_dict

        # change the name of the weights which exists in other model
        # change_dict = {
        #         'conv1.weight':'base.0.weight',
        #         'bn1.running_mean':'base.1.running_mean',
        #         'bn1.running_var':'base.1.running_var',
        #         'bn1.bias':'base.1.bias',
        #         'bn1.weight':'base.1.weight',
        #         }
        # for k, v in list(checkpoint.items()):
        #     for _k, _v in list(change_dict.items()):
        #         if _k == k:
        #             new_key = k.replace(_k, _v)
        #             checkpoint[new_key] = checkpoint.pop(k)
        # change_dict = {'layer1.{:d}.'.format(i):'base.{:d}.'.format(i+4) for i in range(20)}
        # change_dict.update({'layer2.{:d}.'.format(i):'base.{:d}.'.format(i+7) for i in range(20)})
        # change_dict.update({'layer3.{:d}.'.format(i):'base.{:d}.'.format(i+11) for i in range(30)})
        # for k, v in list(checkpoint.items()):
        #     for _k, _v in list(change_dict.items()):
        #         if _k in k:
        #             new_key = k.replace(_k, _v)
        #             checkpoint[new_key] = checkpoint.pop(k)

        resume_scope = self.cfg.TRAIN.RESUME_SCOPE
        # extract the weights based on the resume scope
        if resume_scope != '':
            pretrained_dict = {}
            for k, v in list(checkpoint.items()):
                for resume_key in resume_scope.split(','):
                    if resume_key in k:
                        pretrained_dict[k] = v
                        break
            checkpoint = pretrained_dict

        if torch.cuda.device_count() > 1 and self.multi_gpu:
            torch_model = self.model.module
        else:
            torch_model = self.model

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in torch_model.state_dict()}
        # print("=> Resume weigths:")
        # print([k for k, v in list(pretrained_dict.items())])

        checkpoint = torch_model.state_dict()

        unresume_dict = set(checkpoint)-set(pretrained_dict)
        if len(unresume_dict) != 0:
            print("=> UNResume weigths:")
            print(unresume_dict)

        checkpoint.update(pretrained_dict)

        return torch_model.load_state_dict(checkpoint)

    def resume_checkpoint(self, resume_checkpoint):
        if resume_checkpoint == '' or not os.path.isfile(resume_checkpoint):
            print(("=> no checkpoint found at '{}'".format(resume_checkpoint)))
            return False
        print(("=> loading checkpoint '{:s}'".format(resume_checkpoint)))
        if torch.cuda.device_count() > 1 and self.multi_gpu:
            torch_model = self.model.module
        else:
            torch_model = self.model
        return torch_model.load_state_dict(torch.load(resume_checkpoint))


    def find_previous(self):
        if not os.path.exists(os.path.join(self.output_dir, 'checkpoint_list.txt')):
            return False
        with open(os.path.join(self.output_dir, 'checkpoint_list.txt'), 'r') as f:
            lineList = f.readlines()
        epoches, resume_checkpoints = [list() for _ in range(2)]
        for line in lineList:
            epoch = int(line[line.find('epoch ') + len('epoch '): line.find(':')])
            checkpoint = line[line.find(':') + 2:-1]
            epoches.append(epoch)
            resume_checkpoints.append(checkpoint)
        return epoches, resume_checkpoints

    def weights_init(self, m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0


    def initialize(self):
        # TODO: ADD INIT ways
        # raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")
        # for module in self.cfg.TRAIN.TRAINABLE_SCOPE.split(','):
        #     if hasattr(self.model, module):
        #         getattr(self.model, module).apply(self.weights_init)
        if self.checkpoint:
            print('Loading initial model weights from {:s}'.format(self.checkpoint))
            self.resume_checkpoint_initial(self.checkpoint)

        start_epoch = 0
        return start_epoch

    def trainable_param(self, trainable_scope):
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_param = []

        for module in trainable_scope.split(','):
            if torch.cuda.device_count() > 1 and self.multi_gpu:
                dec = hasattr(self.model.module, module)
                torch_model = self.model.module
            else:
                dec = hasattr(self.model, module)
                torch_model = self.model

            if dec:
                # print(getattr(self.model, module))

                for param in getattr(torch_model, module).parameters():
                    param.requires_grad = True
                trainable_param.extend(getattr(torch_model, module).parameters())

        return trainable_param

    def train_model(self):
        previous = self.find_previous()
        if previous:
            start_epoch = previous[0][-1]
            self.resume_checkpoint(previous[1][-1])
        else:
            start_epoch = self.initialize()

        # export graph for the model, onnx always not works
        # self.export_graph()

        # warm_up epoch
        warm_up = self.cfg.TRAIN.LR_SCHEDULER.WARM_UP_EPOCHS
        for epoch in iter(range(start_epoch+1, self.max_epochs+1)):
            #learning rate
            sys.stdout.write('\rEpoch {epoch:d}/{max_epochs:d}:\n'.format(epoch=epoch, max_epochs=self.max_epochs))
            if epoch > warm_up:
                self.exp_lr_scheduler.step(epoch-warm_up)
            if 'train' in cfg.PHASE:
                self.train_epoch(self.model, self.train_loader, self.optimizer, self.criterion, self.writer, epoch, self.use_gpu)
            if 'eval' in cfg.PHASE:
                with torch.no_grad():
                    self.eval_epoch(self.model, self.eval_loader, self.detector, self.criterion, self.writer, epoch, self.use_gpu)
            if 'test' in cfg.PHASE:
                with torch.no_grad():
                    self.test_epoch(self.model, self.test_loader, self.detector, self.output_dir, self.use_gpu)
            if 'visualize' in cfg.PHASE:
                self.visualize_epoch(self.model, self.visualize_loader, self.priorbox, self.writer, epoch,  self.use_gpu)

            if epoch % cfg.TRAIN.CHECKPOINTS_EPOCHS == 0:
                self.save_checkpoints(epoch)

    def test_model(self):
        previous = self.find_previous()
        if previous:
            for epoch, resume_checkpoint in zip(previous[0], previous[1]):
                if self.cfg.TEST.TEST_SCOPE[0] <= epoch <= self.cfg.TEST.TEST_SCOPE[1]:
                    sys.stdout.write('\rEpoch {epoch:d}/{max_epochs:d}:\n'.format(epoch=epoch, max_epochs=self.cfg.TEST.TEST_SCOPE[1]))
                    self.resume_checkpoint(resume_checkpoint)
                    if 'eval' in cfg.PHASE:
                        self.eval_epoch(self.model, self.eval_loader, self.detector, self.criterion, self.writer, epoch, self.use_gpu)
                    if 'test' in cfg.PHASE:
                        self.test_epoch(self.model, self.test_loader, self.detector, self.output_dir , self.use_gpu)
                    if 'visualize' in cfg.PHASE:
                        self.visualize_epoch(self.model, self.visualize_loader, self.priorbox, self.writer, epoch,  self.use_gpu)
                    if 'custom_visualize' in cfg.PHASE:
                        self.custom_visualize(self.model, self.test_loader, self.detector, self.output_dir,
                                              self.use_gpu)
        else:
            sys.stdout.write('\rCheckpoint {}:\n'.format(self.checkpoint))
            self.resume_checkpoint(self.checkpoint)
            if 'eval' in cfg.PHASE:
                self.eval_epoch(self.model, self.eval_loader, self.detector, self.criterion, self.writer, 0, self.use_gpu)
            if 'test' in cfg.PHASE:
                self.test_epoch(self.model, self.test_loader, self.detector, self.output_dir , self.use_gpu)
            if 'custom_visualize' in cfg.PHASE:
                self.custom_visualize(self.model, self.test_loader, self.detector, self.output_dir , self.use_gpu)
            if 'visualize' in cfg.PHASE:
                self.visualize_epoch(self.model, self.visualize_loader, self.priorbox, self.writer, 0,  self.use_gpu)


    def train_epoch(self, model, data_loader, optimizer, criterion, writer, epoch, use_gpu):
        model.train()

        epoch_size = len(data_loader)
        batch_iterator = iter(data_loader)

        loc_loss = 0
        conf_loss = 0
        _t = Timer()

        for iteration in iter(range((epoch_size))):
            images, targets, idxs, _ = next(batch_iterator)
            if use_gpu:
                images = Variable(images.cuda())
                with torch.no_grad():
                    targets = [Variable(anno.cuda()) for anno in targets]
            else:
                images = Variable(images)
                with torch.no_grad():
                    targets = [Variable(anno) for anno in targets]
            _t.tic()
            # forward
            out = model(images, phase='train')

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)

            # some bugs in coco train2017. maybe the annonation bug.
            if loss_l.item() == float("Inf"):
                continue

            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()

            time = _t.toc()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()



            # log per iter
            log = '\r==>Train: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f}\r'.format(
                    prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
                    time=time, loc_loss=loss_l.item(), cls_loss=loss_c.item())

            del images, targets, loss, loss_l, loss_c, out

            sys.stdout.write(log)
            sys.stdout.flush()

        # log per epoch
        sys.stdout.write('\r')
        sys.stdout.flush()
        lr = optimizer.param_groups[0]['lr']
        log = '\r==>Train: || Total_time: {time:.3f}s || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f} || lr: {lr:.6f}\n'.format(lr=lr,
                time=_t.total_time, loc_loss=loc_loss/epoch_size, conf_loss=conf_loss/epoch_size)
        sys.stdout.write(log)
        sys.stdout.flush()

        # log for tensorboard
        writer.add_scalar('Train/loc_loss', loc_loss/epoch_size, epoch)
        writer.add_scalar('Train/conf_loss', conf_loss/epoch_size, epoch)
        writer.add_scalar('Train/lr', lr, epoch)


    def eval_epoch(self, model, data_loader, detector, criterion, writer, epoch, use_gpu):
        model.eval()

        epoch_size = len(data_loader)
        batch_iterator = iter(data_loader)

        loc_loss = 0
        conf_loss = 0
        _t = Timer()

        if torch.cuda.device_count() > 1 and self.multi_gpu:
            torch_model = model.module
        else:
            torch_model = model

        label = [list() for _ in range(torch_model.num_classes)]
        gt_label = [list() for _ in range(torch_model.num_classes)]
        score = [list() for _ in range(torch_model.num_classes)]
        size = [list() for _ in range(torch_model.num_classes)]
        npos = [0] * torch_model.num_classes

        for iteration in iter(range((epoch_size))):
        # for iteration in iter(range((10))):
            images, targets, idxs, _ = next(batch_iterator)
            if use_gpu:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]

            _t.tic()
            # forward
            out = model(images, phase='train')

            # loss
            loss_l, loss_c = criterion(out, targets)

            out = (out[0], torch_model.softmax(out[1].view(-1, torch_model.num_classes)))

            # detect
            detections = detector.forward(out)

            time = _t.toc()

            # evals
            label, score, npos, gt_label = cal_tp_fp(detections, targets, label, score, npos, gt_label)
            size = cal_size(detections, targets, size)
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()


            # log per iter
            log = '\r==>Eval: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f}\r'.format(
                    prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
                    time=time, loc_loss=loss_l.item(), cls_loss=loss_c.item())

            del images, loss_l, loss_c, out, detections

            sys.stdout.write(log)
            sys.stdout.flush()

        # eval mAP
        prec, rec, ap = cal_pr(label, score, npos)

        # log per epoch
        sys.stdout.write('\r')
        sys.stdout.flush()
        log = '\r==>Eval: || Total_time: {time:.3f}s || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f} || mAP: {mAP:.6f}\n'.format(mAP=ap,
                time=_t.total_time, loc_loss=loc_loss/epoch_size, conf_loss=conf_loss/epoch_size)
        sys.stdout.write(log)
        sys.stdout.flush()

        # log for tensorboard
        writer.add_scalar('Eval/loc_loss', loc_loss/epoch_size, epoch)
        writer.add_scalar('Eval/conf_loss', conf_loss/epoch_size, epoch)
        writer.add_scalar('Eval/mAP', ap, epoch)
        viz_pr_curve(writer, prec, rec, epoch)
        viz_archor_strategy(writer, size, gt_label, epoch)

    # TODO: HOW TO MAKE THE DATALOADER WITHOUT SHUFFLE
    # def test_epoch(self, model, data_loader, detector, output_dir, use_gpu):
    #     # sys.stdout.write('\r===> Eval mode\n')

    #     model.eval()

    #     num_images = len(data_loader.dataset)
    #     num_classes = detector.num_classes
    #     batch_size = data_loader.batch_size
    #     all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    #     empty_array = np.transpose(np.array([[],[],[],[],[]]),(1,0))

    #     epoch_size = len(data_loader)
    #     batch_iterator = iter(data_loader)

    #     _t = Timer()

    #     for iteration in iter(range((epoch_size))):
    #         images, targets = next(batch_iterator)
    #         targets = [[anno[0][1], anno[0][0], anno[0][1], anno[0][0]] for anno in targets] # contains the image size
    #         if use_gpu:
    #             images = Variable(images.cuda())
    #         else:
    #             images = Variable(images)

    #         _t.tic()
    #         # forward
    #         out = model(images, is_train=False)

    #         # detect
    #         detections = detector.forward(out)

    #         time = _t.toc()

    #         # TODO: make it smart:
    #         for i, (dets, scale) in enumerate(zip(detections, targets)):
    #             for j in range(1, num_classes):
    #                 cls_dets = list()
    #                 for det in dets[j]:
    #                     if det[0] > 0:
    #                         d = det.cpu().numpy()
    #                         score, box = d[0], d[1:]
    #                         box *= scale
    #                         box = np.append(box, score)
    #                         cls_dets.append(box)
    #                 if len(cls_dets) == 0:
    #                     cls_dets = empty_array
    #                 all_boxes[j][iteration*batch_size+i] = np.array(cls_dets)

    #         # log per iter
    #         log = '\r==>Test: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}]\r'.format(
    #                 prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
    #                 time=time)
    #         sys.stdout.write(log)
    #         sys.stdout.flush()

    #     # write result to pkl
    #     with open(os.path.join(output_dir, 'detections.pkl'), 'wb') as f:
    #         pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    #     print('Evaluating detections')
    #     data_loader.dataset.evaluate_detections(all_boxes, output_dir)


    def test_epoch(self, model, data_loader, detector, output_dir, use_gpu):
        model.eval()

        epoch_size = len(data_loader)
        batch_iterator = iter(data_loader)

        dataset = data_loader.dataset
        num_images = len(dataset)
        num_classes = detector.num_classes
        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        empty_array = np.transpose(np.array([[],[],[],[],[]]),(1,0))

        _t = Timer()

        # for i in iter(range((num_images))):
        for iteration in iter(range((epoch_size))):
#            # img = dataset.pull_image(i)
#            # scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
            if batch_iterator is not None:
                images, targets, idxs, orig_shape = next(batch_iterator)
            else:
                continue
            scale = [[o[1], o[0], o[1], o[0]] for o in orig_shape]
            # img_dummy = [dataset.pull_image(i) for i in idxs]
#            # scale = [[img.shape[1], img.shape[0], img.shape[1], img.shape[0]] for img in img_dummy]

#            # if use_gpu:
#            #     images = Variable(dataset.preproc(img)[0].unsqueeze(0).cuda(), volatile=True)
#            # else:
#            #     images = Variable(dataset.preproc(img)[0].unsqueeze(0), volatile=True)

            if use_gpu:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
               images = Variable(images)
               targets = [Variable(anno, volatile=True) for anno in targets]

            _t.tic()
#            # forward
            out = model(images, phase='eval')

#            # detect
            detections = detector.forward(out)
            detections = detections.detach().cpu().numpy()
            # boxes, scores = out
            # scores = scores.view(boxes.size(0),-1,scores.size(1))

            # boxes = boxes.detach().cpu().numpy()
            # scores = scores.detach().cpu().numpy()

            time = _t.toc()

            # TODO: make it smart:
            for i, img_id in enumerate(idxs):
                for j in range(1, num_classes):
                    cls_dets = list()
                    cls_mask = np.where(detections[i,j,:,0] > 0)[0]
                    if len(cls_mask) > 0:
                        box = detections[i, j, cls_mask, 1:]
                        box *= np.array(scale[i])
                        score = detections[i, j, cls_mask, 0:1]
                        #
                        # for ik in range(box.shape[0]):
                        #     box[ik, 0] = max(0, box[ik, 0])
                        #     box[ik, 2] = max(min(scale[i][0], box[ik, 2]),box[ik, 0])
                        #     box[ik, 1] = max(0, box[ik, 1])
                        #     box[ik, 3] = max(min(scale[i][1], box[ik, 3]),box[ik, 1])

                        cls_dets = np.concatenate((box, score), axis=1)
#                    #for box,score in zip(boxes_img, scores_img):
#                    #    if score[j] > 0:
#                            # d = det.cpu().numpy()
#                            # score, box = d[0], d[1:]
#                            # box = box
#                            # score = score
#                            #box *= scale[i]
#                            #box = np.append(box, score[j])
#                            #cls_dets.append(box)
                    if len(cls_dets) == 0:
                        cls_dets = empty_array
                    all_boxes[j][img_id] = np.array(cls_dets)

            # im_to_plot = dataset.pull_image(0)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # for label_ind, label in enumerate(dataset._classes[1:]):
            #     for detection in all_boxes[label_ind+1][0]:
            #         box = detection[:4].astype(np.int32)
            #         score = detection[-1]
            #         if (score < 0.3):
            #             continue
            #         else:
            #             print(label, box, score)
            #
            #         cv2.rectangle(im_to_plot, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            #         txt = '{}: {:.3f}'.format(label, score)
            #         cv2.putText(im_to_plot, txt, (box[0], box[1]), font, 0.5, (255, 255, 255), 1)
            # plt.imshow(im_to_plot)
            # plt.show()

            # log per iter
            log = '\r==>Test: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}]\r'.format(
                    prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
                    time=time)
            sys.stdout.write(log)
            sys.stdout.flush()

            del images, out, detections


        # write result to pkl
#        with open(os.path.join(output_dir, 'detections.pkl'), 'wb') as f:
#            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        # currently the COCO dataset do not return the mean ap or ap 0.5:0.95 values
        print('Evaluating detections')
        data_loader.dataset.evaluate_detections(all_boxes, output_dir)

    def custom_visualize(self, model, data_loader, detector, output_dir, use_gpu):
        model.eval()

        dataset = data_loader.dataset
        img_path = '/home/bmudassar3/work/ssds.pytorch/data/Custom/MOT17-13'
        custom_imgList = sorted(glob.glob(img_path + '/*.jpg'))

        num_images = len(custom_imgList)
        num_classes = detector.num_classes
        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

        _t = Timer()

        # for i in iter(range((num_images))):
        for iteration in iter(range((num_images))):
            img = cv2.imread(custom_imgList[iteration], cv2.IMREAD_COLOR)
            scale = [[img.shape[1], img.shape[0], img.shape[1], img.shape[0]]]
            if use_gpu:
                images = Variable(dataset.preproc(img)[0].unsqueeze(0).cuda(), volatile=True)
            else:
                images = Variable(dataset.preproc(img)[0].unsqueeze(0), volatile=True)



            _t.tic()
            #            # forward
            out = model(images, phase='eval')

            #            # detect
            detections = detector.forward(out)
            detections = detections.detach().cpu().numpy()
            # boxes, scores = out
            # scores = scores.view(boxes.size(0),-1,scores.size(1))

            # boxes = boxes.detach().cpu().numpy()
            # scores = scores.detach().cpu().numpy()

            time = _t.toc()

            # TODO: make it smart:
            for j in range(1, 2):
                cls_dets = list()
                cls_mask = np.where(detections[0, j, :, 0] > 0.3)[0]
                if len(cls_mask) > 0:
                    box = detections[0, j, cls_mask, 1:]
                    box *= np.array(scale[0])
                    score = detections[0, j, cls_mask, 0:1]
                    cls_dets = np.concatenate((box, score), axis=1)

                if len(cls_dets) == 0:
                    cls_dets = empty_array
                all_boxes[j][0] = np.array(cls_dets)

            im_to_plot = img
            font = cv2.FONT_HERSHEY_SIMPLEX
            for label_ind, label in enumerate(dataset._classes[1:2]):
                for detection in all_boxes[label_ind+1][0]:
                    box = detection[:4].astype(np.int32)
                    score = detection[-1]
                    if (score < 0.3):
                        continue
                    else:
                        print(label, box, score)

                    cv2.rectangle(im_to_plot, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                    txt = '{}: {:.3f}'.format(label, score)
                    cv2.putText(im_to_plot, txt, (box[0], box[1]), font, 0.5, (255, 255, 255), 1)
            # plt.imshow(im_to_plot)
            # plt.show()
            cv2.imwrite(img_path+'/results/'+str(iteration).zfill(6)+'.jpg', im_to_plot)

            # log per iter



    def visualize_epoch(self, model, data_loader, priorbox, writer, epoch, use_gpu):
        model.eval()

        img_index = random.randint(0, len(data_loader.dataset)-1)

        # get img
        image = data_loader.dataset.pull_image(img_index)
        anno = data_loader.dataset.pull_anno(img_index)

        # visualize archor box
        viz_prior_box(writer, priorbox, image, epoch)

        # get preproc
        preproc = data_loader.dataset.preproc
        preproc.add_writer(writer, epoch)
        # preproc.p = 0.6

        # preproc image & visualize preprocess prograss
        images = Variable(preproc(image, anno)[0].unsqueeze(0), volatile=True)
        if use_gpu:
            images = images.cuda()

        # visualize feature map in base and extras
        base_out = viz_module_feature_maps(writer, model.base, images, module_name='base', epoch=epoch)
        extras_out = viz_module_feature_maps(writer, model.extras, base_out, module_name='extras', epoch=epoch)
        # visualize feature map in feature_extractors
        viz_feature_maps(writer, model(images, 'feature'), module_name='feature_extractors', epoch=epoch)

        model.train()
        images.requires_grad = True
        images.volatile=False
        base_out = viz_module_grads(writer, model, model.base, images, images, preproc.means, module_name='base', epoch=epoch)

        # TODO: add more...


    def configure_optimizer(self, trainable_param, cfg):
        if cfg.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(trainable_param, lr=cfg.LEARNING_RATE,
                        momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
        elif cfg.OPTIMIZER == 'rmsprop':
            optimizer = optim.RMSprop(trainable_param, lr=cfg.LEARNING_RATE,
                        momentum=cfg.MOMENTUM, alpha=cfg.MOMENTUM_2, eps=cfg.EPS, weight_decay=cfg.WEIGHT_DECAY)
        elif cfg.OPTIMIZER == 'adam':
            optimizer = optim.Adam(trainable_param, lr=cfg.LEARNING_RATE,
                        betas=(cfg.MOMENTUM, cfg.MOMENTUM_2), eps=cfg.EPS, weight_decay=cfg.WEIGHT_DECAY)
        else:
            AssertionError('optimizer can not be recognized.')
        return optimizer


    def configure_lr_scheduler(self, optimizer, cfg):
        if cfg.SCHEDULER == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.STEPS[0], gamma=cfg.GAMMA)
        elif cfg.SCHEDULER == 'multi_step':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.STEPS, gamma=cfg.GAMMA)
        elif cfg.SCHEDULER == 'exponential':
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg.GAMMA)
        elif cfg.SCHEDULER == 'SGDR':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.MAX_EPOCHS)
        else:
            AssertionError('scheduler can not be recognized.')
        return scheduler


    def export_graph(self):
        self.model.train(False)
        dummy_input = Variable(torch.randn(1, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])).cuda()
        # Export the model
        torch_out = torch.onnx._export(self.model,             # model being run
                                       dummy_input,            # model input (or a tuple for multiple inputs)
                                       "graph.onnx",           # where to save the model (can be a file or file-like object)
                                       export_params=True)     # store the trained parameter weights inside the model file
        # if not os.path.exists(cfg.EXP_DIR):
        #     os.makedirs(cfg.EXP_DIR)
        # self.writer.add_graph(self.model, (dummy_input, ))


def train_model(args):
    # try:
    s = Solver(args)
    s.train_model()
    # except RuntimeError:
    #     pass
    return True

def test_model(args):
    s = Solver(args)
    s.test_model()
    return True
