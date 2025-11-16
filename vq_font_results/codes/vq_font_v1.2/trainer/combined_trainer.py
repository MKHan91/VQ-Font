
from .base_trainer import BaseTrainer
import utils
import json
# from basicsr.utils import USMSharp
def is_main_worker(gpu):
    return (gpu <= 0)
import torch

class CombinedTrainer(BaseTrainer):
    """
    CombinedTrainer
    """
    def __init__(self, ddp_gpu,gen, disc, g_optim, d_optim, g_scheduler, d_scheduler,
                 logger, evaluator, cv_loaders, cfg, writer): # cls_char
        super().__init__(ddp_gpu,gen, disc, g_optim, d_optim, g_scheduler, d_scheduler,
                         logger, evaluator, cv_loaders, cfg, writer)

    # region - train
    def train(self, loader, st_step=1, max_step=100000):
        self.max_step = max_step
        self.num_epoch = int(max_step // len(loader))
        
        self.gen.train()
        if self.disc is not None:
            self.disc.train()

        # loss stats
        losses = utils.AverageMeters("g_total", "pixel", "disc", "gen","lpips","cross","l1","feat")
        # discriminator stats
        discs = utils.AverageMeters("real_font", "real_uni", "fake_font", "fake_uni")
        # etc stats
        stats = utils.AverageMeters("B_style", "B_target")
        self.step = st_step
        self.clear_losses()
        self.logger.info("Start training ...")
        
        with open('/home/dev/VQ-Font/build_dataset/structure_tags.json','r') as f:
            stru_map = json.load(f,strict=False)
        with open('/home/dev/VQ-Font/build_dataset/cr_mapping.json','r') as f:
            cr_map = json.load(f,strict=False)
        with open('/home/dev/VQ-Font/build_dataset/de.json','r') as f:
            de = json.load(f,strict=False)
            
        while True:
            for (in_style_ids, in_imgs,in_imgs_ske,
                 trg_style_ids, trg_uni_ids, trg_imgs, content_imgs, content_imgs_ske,trg_unis, style_sample_index, trg_sample_index) in loader:
                
                self.epoch = self.step // len(loader)
                B = trg_imgs.shape[0]
                stats.updates({
                    "B_style": in_imgs.shape[0],#batch*k_shot
                    "B_target": B#batch
                })
                
                in_style_ids = in_style_ids.cuda()
                in_imgs = in_imgs.cuda()        
                trg_uni_disc_ids = trg_uni_ids.cuda()
                trg_style_ids = trg_style_ids.cuda()
                trg_imgs = trg_imgs.cuda()
                content_imgs = content_imgs.cuda()
                
                #获取in_styles_unis
                in_styles_unis = []
                for i in trg_unis:
                    in_styles_unis.append(cr_map[i[0]])

                #获取结构信息
                trg_stru_ids = []
                for i in trg_unis:
                    trg_stru_ids.append(stru_map[i[0]])
                trg_stru_ids = torch.tensor(trg_stru_ids).cuda()
                
                in_stru_ids=[]
                for k in in_styles_unis:
                    for i in range(3):
                        in_stru_ids.append(stru_map[k[i]])
                in_stru_ids = torch.tensor(in_stru_ids).cuda()

                #获取部件信息
                trg_comp_ids = []
                for i in trg_unis:
                    trg_comp_ids.append(de[i[0]])
                in_comp_ids=[]
                for k in in_styles_unis:
                    for i in range(3):
                        in_comp_ids.append(de[k[i]])
                
                if self.cfg.use_half:
                    in_imgs = in_imgs.half()
                    content_imgs = content_imgs.half()

                in_imgs_crose = torch.nn.functional.interpolate(in_imgs,scale_factor=1.2,mode='bilinear')
                in_imgs_crose = in_imgs_crose.cuda()
                in_imgs_fine = torch.nn.functional.interpolate(in_imgs,scale_factor=0.8,mode='bilinear')
                in_imgs_fine = in_imgs_fine.cuda()          
                trg_imgs_crose = torch.nn.functional.interpolate(trg_imgs,scale_factor=1.2,mode='bilinear')
                trg_imgs_fine = torch.nn.functional.interpolate(trg_imgs,scale_factor=0.8,mode='bilinear')
                
                ##############################################################
                # infer
                ##############################################################
                # quant, emb_loss, info ,gt_feat= self.gen.vqgan.encode(trg_imgs) #info[2]:[2048]
                quant, emb_loss, info = self.gen.vqgan.encode(trg_imgs) #info[2]:[2048]
                tar = self.gen.vqgan.decode(quant)
                sc_feats = self.gen.encode_write_comb(in_style_ids, style_sample_index, in_imgs,in_imgs_crose,in_imgs_fine,in_stru_ids)
                out, z_e_x,_,z_q_x ,indice_out= self.gen.read_decode(trg_style_ids, trg_sample_index, content_imgs,trg_stru_ids,in_stru_ids) #fake_img
                self_infer_imgs, z_e_x_self ,_,z_q_x_self,indice_self= self.gen.infer(trg_style_ids, trg_imgs,trg_imgs_crose,trg_imgs_fine, trg_style_ids, trg_sample_index, trg_sample_index, content_imgs,trg_stru_ids,trg_stru_ids)
                
                ################### discriminator ##################
                real_font, real_uni, real_stru= self.disc(trg_imgs, trg_style_ids, trg_uni_disc_ids,trg_stru_ids)
                fake_font, fake_uni,fake_stru = self.disc(out.detach(), trg_style_ids, trg_uni_disc_ids,trg_stru_ids)
                self.add_gan_d_loss(real_stru, real_uni, fake_stru,fake_uni)
                self.d_optim.zero_grad()
                self.d_backward()
                self.d_optim.step()
                self.d_scheduler.step()

                ################### generator ##################
                fake_font, fake_uni,fake_stru = self.disc(out, trg_style_ids, trg_uni_disc_ids,trg_stru_ids)
                self.add_gan_g_loss(real_stru, real_stru, fake_uni, fake_stru)
                self.add_l1_loss_only_mainstructure(out, trg_imgs)
                self.add_lpips_loss_only_mainstructure(out, trg_imgs)
                self.add_crossentropy_loss(indice_out, info[2], indice_self)
                self.g_optim.zero_grad()
                self.g_backward()
                self.g_optim.step()
                self.g_scheduler.step()
                loss_dic = self.clear_losses()
                losses.updates(loss_dic, B)  # accum loss stats

                # EMA g
                self.accum_g()
                if self.step % self.cfg['tb_freq'] == 0:
                    tag_scalar_dic = self.baseplot(losses, discs, stats)

                if self.step % self.cfg['print_freq'] == 0:
                    self.log(losses, discs, stats)
                    losses.resets()
                    discs.resets()
                    stats.resets()

                if self.step % self.cfg['val_freq'] == 0:
                    if is_main_worker(self.ddp_gpu):
                        epoch = self.step / len(loader)
                        self.logger.info(f"Validation at Epoch = {epoch:.3f}")
                        self.evaluator.cp_validation(self.gen_ema, self.cv_loaders, self.step)
                        
                        self.writer.add_scalars({"learning_rate/generator": self.g_optim.param_groups[0]['lr']}, self.step)
                        self.writer.add_scalars({"learning_rate/discriminator": self.d_optim.param_groups[0]['lr']}, self.step)
                        
                # if self.step >= 250000 and self.step % self.cfg['val_freq']==0:     
                if self.step >= 25000 and self.step % self.cfg['val_freq']==0:     
                    self.save(loss_dic['g_total'], self.cfg['save'], self.cfg.get('save_freq', self.cfg['val_freq']))
                    
                if self.step >= max_step:
                    break

                self.step += 1
                
            if self.step >= max_step:
                break
            
        self.logger.info("Iteration finished.")

    
    # region - log
    def log(self, losses, discs, stats):
        self.logger.info(
            f" Epoch: [{self.epoch:4d}/{self.num_epoch}]  Step: {self.step:7d} "
            f" | Cross: {losses.cross.avg:7.4f},  L1: {losses.l1.avg:7.4f},  Lpips: {losses.lpips.avg:7.4f},  Feat: {losses.feat.avg:7.4f},  D: {losses.disc.avg:7.3f},  G: {losses.gen.avg:7.3f}"
            f" | B_stl: {stats.B_style.avg:5.1f},  B_trg: {stats.B_target.avg:5.1f}"
            )
