import logging
import os
import wave
import io
import soundfile
from pydub import AudioSegment
import librosa

from inference import infer_tool
from inference.infer_tool import Svc
from spkmix import spk_mix_map

import subprocess
import edge_tts
import asyncio
import argparse

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")

class TTS():
    def __init__(self, args):
        self.clean_names = args['clean_names']
        self.trans = args['trans']
        self.spk_list = args['spk_list']
        self.slice_db = args['slice_db']
        self.wav_format = args['wav_format']
        self.auto_predict_f0 = args['auto_predict_f0']
        self.cluster_infer_ratio = args['cluster_infer_ratio']
        self.noice_scale = args['noice_scale']
        self.pad_seconds = args['pad_seconds']
        self.clip = args['clip']
        self.lg = args['linear_gradient']
        self.lgr = args['linear_gradient_retain']
        self.f0p = args['f0_predictor']
        self.enhance = args['enhance']
        self.enhancer_adaptive_key = args['enhancer_adaptive_key']
        self.cr_threshold = args['f0_filter_threshold']
        self.diffusion_model_path = args['diffusion_model_path']
        self.diffusion_config_path = args['diffusion_config_path']
        self.k_step = args['k_step']
        self.only_diffusion = args['only_diffusion']
        self.shallow_diffusion = args['shallow_diffusion']
        self.use_spk_mix = args['use_spk_mix']
        self.second_encoding = args['second_encoding']
        self.loudness_envelope_adjustment = args['loudness_envelope_adjustment']
        self.edge_tts_rate = args['edge_tts_rate']
        self.edge_tts_voice = args['edge_tts_voice']

        self.svc_model = Svc(args['model_path'],
                            args['config_path'],
                            args['device'],
                            args['cluster_model_path'],
                            args['enhance'],
                            args['diffusion_model_path'],
                            args['diffusion_config_path'],
                            args['shallow_diffusion'],
                            args['only_diffusion'],
                            args['use_spk_mix'],
                            args['feature_retrieval'])
        
    
    #voice = self.edge_tts_voice
    #rate = self.edge_tts_rate  
    def tts_func(self,_text, _rate= 0, _voice= "男"):
            #使用edge-tts把文字转成音频
            # voice = "zh-CN-XiaoyiNeural"#女性，较高音
            # voice = "zh-CN-YunxiNeural"#男性
            voice = "zh-CN-YunxiNeural"#男性
            if ( _voice == "女" ) :
                voice = "zh-CN-XiaoyiNeural"
            #output_file = "C:/Users/zhangkaihao/so-vits-svc/raw/"+_text[0:10]+".wav"
            output_file = "C:/Users/zhangkaihao/so-vits-svc-modified/raw/audio.wav"
            # communicate = edge_tts.Communicate(_text, voice)
            # await communicate.save(output_file)
            if _rate>=0:
                ratestr="+{:.0%}".format(_rate)
            elif _rate<0:
                ratestr="{:.0%}".format(_rate)#减号自带

            if os.path.exists(f'C:/Users/zhangkaihao/so-vits-svc-modified/raw/audio.wav'):
                os.remove(f'C:/Users/zhangkaihao/so-vits-svc-modified/raw/audio.wav')
            p=subprocess.Popen("edge-tts "+
                                " --text "+_text+
                                " --write-media "+output_file+
                                " --voice "+voice+
                                " --rate="+ratestr
                                ,shell=True,
                                stdout=subprocess.PIPE,
                                stdin=subprocess.PIPE)
            p.wait()
            #return output_file


    def model(self):
        svc_model = self.svc_model

        infer_tool.mkdir(["raw", "results"])

        if len(spk_mix_map)<=1:
            self.use_spk_mix = False
        if self.use_spk_mix:
            self.spk_list = [spk_mix_map]

        infer_tool.fill_a_to_b(self.trans, self.clean_names)
        for clean_name, tran in zip(self.clean_names, self.trans):
            raw_audio_path = f"raw/{clean_name}"
            if "." not in raw_audio_path:
                raw_audio_path += ".wav"
            infer_tool.format_wav(raw_audio_path)
            for spk in self.spk_list:
                kwarg = {
                    "raw_audio_path" : raw_audio_path,
                    "spk" : spk,
                    "tran" : tran,
                    "slice_db" : self.slice_db,
                    "cluster_infer_ratio" : self.cluster_infer_ratio,
                    "auto_predict_f0" : self.auto_predict_f0,
                    "noice_scale" : self.noice_scale,
                    "pad_seconds" : self.pad_seconds,
                    "clip_seconds" : self.clip,
                    "lg_num": self.lg,
                    "lgr_num" : self.lgr,
                    "f0_predictor" : self.f0p,
                    "enhancer_adaptive_key" : self.enhancer_adaptive_key,
                    "cr_threshold" : self.cr_threshold,
                    "k_step": self.k_step,
                    "use_spk_mix": self.use_spk_mix,
                    "second_encoding": self.second_encoding,
                    "loudness_envelope_adjustment": self.loudness_envelope_adjustment
                }
                audio = svc_model.slice_inference(**kwarg)
                key = "auto" if self.auto_predict_f0 else f"{tran}key"
                cluster_name = "" if self.cluster_infer_ratio == 0 else f"_{self.cluster_infer_ratio}"
                isdiffusion = "sovits"
                if self.shallow_diffusion :
                    isdiffusion = "sovdiff"
                if self.only_diffusion :
                    isdiffusion = "diff"
                if self.use_spk_mix:
                    spk = "spk_mix"
                #res_path = f'results/{clean_name}_{key}_{spk}{cluster_name}_{isdiffusion}_{f0p}.{wav_format}'
                if os.path.exists(f'C:/Users/zhangkaihao/so-vits-svc-modified/results/audio.wav'):
                    os.remove(f'c:/Users/zhangkaihao/so-vits-svc-modified/results/audio.wav')
                res_path = f'c:/Users/zhangkaihao/so-vits-svc-modified/results/audio.wav'
                #print(svc_model.target_sample)

                # 重新采样
                target_sampling_rate = 16000
                resampled_audio = librosa.resample(audio, orig_sr=svc_model.target_sample, target_sr=target_sampling_rate)
                soundfile.write(res_path, resampled_audio, target_sampling_rate, format=self.wav_format)
                #soundfile.write(res_path, audio, svc_model.target_sample, format=self.wav_format)
                #soundfile.write(res_path, audio, 16000, format=self.wav_format)

                # 加载你的音频文件（可以是WAV、MP3等格式）
                audio_file = AudioSegment.from_file(res_path, format="wav")

                # 设置目标比特率
                target_bitrate = "32k"

                # 导出音频文件为32kbps比特率
                audio_file.export(res_path, format="wav", bitrate=target_bitrate)



                # # 创建一个 BytesIO 对象以在内存中处理二进制文件
                # wav_binary_data = io.BytesIO()

                # # 保存 NumPy 数组为 WAV 文件
                # with wave.open(wav_binary_data, 'wb') as wav_file:
                #     wav_file.setnchannels(1)  # 设置通道数，例如: 单声道
                #     wav_file.setsampwidth(2)  # 设置采样宽度，例如: 2 (16-bit)
                #     wav_file.setframerate(svc_model.target_sample)  # 设置采样率
                #     wav_file.writeframes(audio.tobytes())  # 将音频帧写入文件
                
                # # 获取二进制数据
                # binary_content = wav_binary_data.getvalue()
                # print(binary_content)

                with open(res_path, 'rb') as f:
                    binary_content = f.read()

                print("Done!")
                svc_model.clear_empty()

                return binary_content

    def final_API_cmd(self):
        import argparse
        parser = argparse.ArgumentParser(description='sovits4 inference')
        parser.add_argument('-txt', '--edge_tts_text', type=str, default='请输入您的文本', help='文本输入')
        args = parser.parse_args()

        text = args.edge_tts_text
        self.tts_func(text)
        self.model()
    
    def final_API(self, text):
        self.tts_func(text)
        bin_data = self.model()
        return bin_data
    
    def process_args(self):
        if self.cluster_infer_ratio != 0:
            if self.cluster_model_path == "":
                if self.feature_retrieval:  # 若指定了占比但没有指定模型路径，则按是否使用特征检索分配默认的模型路径
                    self.cluster_model_path = "logs/44k/feature_and_index.pkl"
                else:
                    self.cluster_model_path = "logs/44k/kmeans_10000.pt"
        else:  # 若未指定占比，则无论是否指定模型路径，都将其置空以避免之后的模型加载
            self.cluster_model_path = ""

""" def parse_arguments():

    parser = argparse.ArgumentParser(description='sovits4 inference')

    # 一定要设置的部分
    parser.add_argument('-m', '--model_path', type=str, default="logs/44k/G_36800.pth", help='模型路径')
    parser.add_argument('-c', '--config_path', type=str, default="logs/44k/config.json", help='配置文件路径')
    parser.add_argument('-cl', '--clip', type=float, default=0, help='音频强制切片，默认0为自动切片，单位为秒/s')
    parser.add_argument('-n', '--clean_names', type=str, nargs='+', default=["audio.wav"], help='wav文件名列表，放在raw文件夹下')
    parser.add_argument('-t', '--trans', type=int, nargs='+', default=[0], help='音高调整，支持正负（半音）')
    parser.add_argument('-s', '--spk_list', type=str, nargs='+', default=['rjdy'], help='合成目标说话人名称')

    # 可选项部分
    parser.add_argument('-a', '--auto_predict_f0', action='store_true', default=False, help='语音转换自动预测音高，转换歌声时不要打开这个会严重跑调')
    parser.add_argument('-cm', '--cluster_model_path', type=str, default="", help='聚类模型或特征检索索引路径，留空则自动设为各方案模型的默认路径，如果没有训练聚类或特征检索则随便填')
    parser.add_argument('-cr', '--cluster_infer_ratio', type=float, default=0, help='聚类方案或特征检索占比，范围0-1，若没有训练聚类模型或特征检索则默认0即可')
    parser.add_argument('-lg', '--linear_gradient', type=float, default=0, help='两段音频切片的交叉淡入长度，如果强制切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，单位为秒')
    parser.add_argument('-f0p', '--f0_predictor', type=str, default="pm", help='选择F0预测器,可选择crepe,pm,dio,harvest,默认为pm(注意：crepe为原F0使用均值滤波器)')
    parser.add_argument('-eh', '--enhance', action='store_true', default=False, help='是否使用NSF_HIFIGAN增强器,该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭')
    parser.add_argument('-shd', '--shallow_diffusion', action='store_true', default=False, help='是否使用浅层扩散，使用后可解决一部分电音问题，默认关闭，该选项打开时，NSF_HIFIGAN增强器将会被禁止')
    parser.add_argument('-usm', '--use_spk_mix', action='store_true', default=False, help='是否使用角色融合')
    parser.add_argument('-lea', '--loudness_envelope_adjustment', type=float, default=1, help='输入源响度包络替换输出响度包络融合比例，越靠近1越使用输出响度包络')
    parser.add_argument('-fr', '--feature_retrieval', action='store_true', default=False, help='是否使用特征检索，如果使用聚类模型将被禁用，且cm与cr参数将会变成特征检索的索引路径与混合比例')

    # 浅扩散设置
    parser.add_argument('-dm', '--diffusion_model_path', type=str, default="logs/44k/diffusion/model_0.pt", help='扩散模型路径')
    parser.add_argument('-dc', '--diffusion_config_path', type=str, default="logs/44k/diffusion/config.yaml", help='扩散模型配置文件路径')
    parser.add_argument('-ks', '--k_step', type=int, default=100, help='扩散步数，越大越接近扩散模型的结果，默认100')
    parser.add_argument('-se', '--second_encoding', action='store_true', default=False, help='二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，有时候效果好，有时候效果差')
    parser.add_argument('-od', '--only_diffusion', action='store_true', default=False, help='纯扩散模式，该模式不会加载sovits模型，以扩散模型推理')


    # 不用动的部分
    parser.add_argument('-sd', '--slice_db', type=int, default=-40, help='默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50')
    parser.add_argument('-d', '--device', type=str, default=None, help='推理设备，None则为自动选择cpu和gpu')
    parser.add_argument('-ns', '--noice_scale', type=float, default=0.4, help='噪音级别，会影响咬字和音质，较为玄学')
    parser.add_argument('-p', '--pad_seconds', type=float, default=0.5, help='推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现')
    parser.add_argument('-wf', '--wav_format', type=str, default='wav', help='音频输出格式')
    parser.add_argument('-lgr', '--linear_gradient_retain', type=float, default=0.75, help='自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭')
    parser.add_argument('-eak', '--enhancer_adaptive_key', type=int, default=0, help='使增强器适应更高的音域(单位为半音数)|默认为0')
    parser.add_argument('-ft', '--f0_filter_threshold', type=float, default=0.05,help='F0过滤阈值，只有使用crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音')

    parser.add_argument('-voi', '--edge_tts_voice', type=str, default='zh-CN-YunxiNeural', help='选择edge_tts声音')
    parser.add_argument('-eot', '--edge_tts_output_path', type=str, default='C:/Users/zhangkaihao/so-vits-svc-modified/raw/audio.wav', help='edge_tts语音保存路径')
    parser.add_argument('-rat', '--edge_tts_rate', type=float, default= 0, help='edge_tts语速')
    parser.add_argument('-vol', '--edge_tts_volume', type=int, default= 0 , help='edge_tts音量')
    parser.add_argument('-pit', '--edge_tts_pitch', type=str, default='0%', help='edge_tts音调')
    parser.add_argument('-txt', '--edge_tts_text', type=str, default='请输入您的文本', help='文本输入')


    args = parser.parse_args()
    return args """