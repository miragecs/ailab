import threading,cv2,torch,os
from random import uniform
from multiprocessing import Queue
import multiprocessing
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from moviepy.editor import VideoFileClip
from upcunet_v3 import RealWaifuUpScaler
from time import time as ttime,sleep
import os,sys
root_path=os.path.abspath('.')
sys.path.append(root_path)
class UpScalerMT(threading.Thread):
    def __init__(self, inp_q, res_q, device, model,p_sleep,nt,tile,cache_mode,alpha):
        threading.Thread.__init__(self)
        self.device = device
        self.inp_q = inp_q
        self.res_q = res_q
        self.model = model
        self.cache_mode=cache_mode
        self.nt = nt
        self.p_sleep=p_sleep
        self.tile=tile
        self.alpha = alpha

    def inference(self, tmp):
        idx, np_frame = tmp
        with torch.no_grad():
            res = self.model(np_frame,self.tile,self.cache_mode,self.alpha)
        if(self.nt>1):
            sleep(uniform(self.p_sleep[0],self.p_sleep[1]))
        return (idx, res)

    def run(self):
        while (1):
            tmp = self.inp_q.get()
            if (tmp == None):
                # print("exit")
                break
            self.res_q.put(self.inference(tmp))
class VideoRealWaifuUpScaler(object):
    def __init__(self,nt,n_gpu,scale,half,tile,cache_mode,alpha,p_sleep,decode_sleep,encode_params,n_cache):
        self.nt = nt
        self.n_gpu = n_gpu  # 每块GPU开nt个进程
        self.scale = scale
        self.encode_params = encode_params
        self.decode_sleep=decode_sleep
        self.n_cache = n_cache
        
        device_base = "cuda"
        self.inp_q = Queue(self.n_cache * self.nt * self.n_gpu * 2)  # 抽帧缓存上限帧数
        self.res_q = Queue(self.n_cache * self.nt * self.n_gpu * 2)  # 超分帧结果缓存上限
        for i in range(self.n_gpu):
            device = device_base + ":%s" % i
            #load+device初始化好当前卡的模型
            model=RealWaifuUpScaler(self.scale, eval("model_path%s" % self.scale), half, device)
            for _ in range(self.nt):
                upscaler = UpScalerMT(self.inp_q, self.res_q, device, model,p_sleep,self.nt,tile,cache_mode,alpha)
                upscaler.start()

    def __call__(self, inp_path,opt_path,tmp_path):
        objVideoreader = VideoFileClip(filename=inp_path)
        w,h=objVideoreader.reader.size
        fps=objVideoreader.reader.fps
        total_frame=objVideoreader.reader.nframes
        if_audio=objVideoreader.audio
        if(if_audio):
            tmp_audio_path="%s.m4a"%tmp_path
            objVideoreader.audio.write_audiofile(tmp_audio_path,codec="aac")
            writer = FFMPEG_VideoWriter(opt_path, (w * self.scale, h * self.scale), fps, ffmpeg_params=self.encode_params,audiofile=tmp_audio_path)  # slower#medium
        else:
            writer = FFMPEG_VideoWriter(opt_path, (w * self.scale, h * self.scale), fps, ffmpeg_params=self.encode_params)  # slower#medium
        now_idx = 0
        idx2res = {}
        t0 = ttime()
        res_idx = 0
        
        while(res_idx < total_frame):
            flag = 0
            for idx, frame in enumerate(objVideoreader.iter_frames()):
                # if idx>=256:
                #     print("当前帧数：%s"%idx)
                if idx >= res_idx:
                    # print(1,idx, self.inp_q.qsize(), self.res_q.qsize(), now_idx, sorted(idx2res.keys()))  ##
                    if(idx%10==0):
                        print("total frame:%s\tdecoded frames:%s"%(int(total_frame),idx))  ##
                    # print("inp_q:%s\t"%self.inp_q.qsize())
                    # print("res_q:%s\t"%self.res_q.qsize())
                    # print("idx2res:%s\t"%sys.getsizeof(idx2res))
                    self.inp_q.put((idx, frame))
                    sleep(self.decode_sleep)#否则解帧会一直抢主进程的CPU到100%，不给其他线程CPU空间进行图像预处理和后处理
                    while (1):  # 取出处理好的所有结果
                        if (self.res_q.empty()): break
                        iidx, res = self.res_q.get()
                        idx2res[iidx] = res
                    # if (idx % 100 == 0):
                    while (1):  # 按照idx排序写帧
                        if (now_idx not in idx2res): break
                        writer.write_frame(idx2res[now_idx])
                        del idx2res[now_idx]
                        now_idx += 1
                    if idx >= min(res_idx + self.nt * self.n_gpu * 2,total_frame): 
                        flag=1
                        break
            idx+=1
            res_idx = idx
            while (1):
                # print(2,idx, self.inp_q.qsize(), self.res_q.qsize(), now_idx, sorted(idx2res.keys()))  ##
                # if (now_idx >= idx + 1): break  # 全部帧都写入了，跳出
                while (1):  # 取出处理好的所有结果
                    if (self.res_q.empty()): break
                    iidx, res = self.res_q.get()
                    idx2res[iidx] = res
                while (1):  # 按照idx排序写帧
                    if (now_idx not in idx2res): break
                    writer.write_frame(idx2res[now_idx])
                    del idx2res[now_idx]
                    now_idx += 1
                if(self.inp_q.qsize()==0 and self.res_q.qsize()==0 and idx==now_idx):break
                sleep(0.02)
            if flag == 0:
                break
        # print(3,idx, self.inp_q.qsize(), self.res_q.qsize(), now_idx, sorted(idx2res.keys()))  ##
        for _ in range(self.nt * self.n_gpu):  # 全部结果拿到后，关掉模型线程
            self.inp_q.put(None)
        writer.close()
        if(if_audio):
            os.remove(tmp_audio_path)
        t1 = ttime()
        print(inp_path,"done,time cost:",t1 - t0)

if __name__ == '__main__':
    #from config import half, model_path2, model_path3, model_path4, tile, scale, device, encode_params, p_sleep, decode_sleep, nt, n_gpu,cache_mode,alpha,inp_path,opt_path
    import argparse

    # 创建解析器
    parser = argparse.ArgumentParser(description="超分视频和图像设置")
    
    # 通用设置
    parser.add_argument("--half", type=bool, help="是否开启半精度计算（True 或 False）")
    parser.add_argument("--model_path2", type=str, help="2倍超分模型的路径")
    parser.add_argument("--model_path3", type=str, help="3倍超分模型的路径")
    parser.add_argument("--model_path4", type=str, help="4倍超分模型的路径")
    parser.add_argument("--tile", type=int, help="瓦片大小")
    parser.add_argument("--scale", type=int, help="超分倍率")
    parser.add_argument("--device", type=str, help="计算设备（'cpu' 或 'cuda:0'）")
    parser.add_argument("--cache_mode", type=int, help="缓存模式")
    parser.add_argument("--alpha", type=float, help="修复程度")
    
    # 超图像设置
    parser.add_argument("--input_dir", type=str, help="输入图像文件夹路径")
    parser.add_argument("--output_dir", type=str, help="超分图像输出文件夹路径")
    
    # 超视频设置
    parser.add_argument("--inp_path", type=str, help="输入视频路径")
    parser.add_argument("--opt_path", type=str, help="输出视频路径")
    parser.add_argument("--nt", type=int, help="线程数")
    parser.add_argument("--n_gpu", type=int, help="显卡数")
    parser.add_argument("--n_cache", type=int, help="队列倍数")
    
    # 解析参数
    args = parser.parse_args()
    
    # 获取参数值
    half = args.half
    model_path2 = args.model_path2
    model_path3 = args.model_path3
    model_path4 = args.model_path4
    tile = args.tile
    scale = args.scale
    device = args.device
    nt = args.nt
    n_gpu = args.n_gpu
    cache_mode = args.cache_mode
    alpha = args.alpha
    inp_path = args.inp_path
    opt_path = args.opt_path
    input_dir = args.input_dir
    output_dir = args.output_dir
    n_cache = args.n_cache
    
    
    p_sleep=(0.005,0.012)
    decode_sleep=0.002
    #编码参数，不懂别乱动;通俗来讲，crf变低=高码率高质量，slower=低编码速度高质量+更吃CPU，CPU不够应该调低级别，比如slow，medium，fast，faster
    encode_params=['-crf', '21', '-preset', 'medium']
    #if inp_path == '':
    #    inp_path = "/content/drive/MyDrive/CUGAN/ailab/Real-CUGAN/inputs/1.mp4"
    #if opt_path == '':
    #    opt_path = "/content/drive/MyDrive/CUGAN/ailab/Real-CUGAN/input_dir/1.mp4"
    os.makedirs("%s/tmp_video"%root_path,exist_ok=True)
    os.makedirs("%s/%s"% (root_path, opt_path),exist_ok=True)
    for name in os.listdir(inp_path):
        
        tmp_path = "%s/tmp_video/%s"% (root_path , name)
        tmp_inp_path = "%s/%s"% (inp_path, name)
        tmp_opt_path = "%s/%s"% (opt_path, name)
        print(tmp_path)
        print(tmp_inp_path)
        print(tmp_opt_path)
        video_upscaler=VideoRealWaifuUpScaler(nt,n_gpu,scale,half,tile,cache_mode,alpha,p_sleep,decode_sleep,encode_params,n_cache)
        video_upscaler(tmp_inp_path,tmp_opt_path,tmp_path)
        
