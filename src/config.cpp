#include "config.h"

#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>

namespace prkd::cfg
{
    namespace
    {
        std::string trim(const std::string& s)
        {
            const auto is_ws = [](unsigned char c) { return std::isspace(c) != 0; };
            size_t b = 0;
            size_t e = s.size();
            while (b < e && is_ws(static_cast<unsigned char>(s[b]))) ++b;
            while (e > b && is_ws(static_cast<unsigned char>(s[e - 1]))) --e;
            return s.substr(b, e - b);
        }

        // 要求【整个】字符串都是整数 —— 否则 "12abc" 会被 stoll 静默截成 12
        int64_t parse_int(const std::string& v, const std::string& where)
        {
            if (v.empty()) throw std::runtime_error(where + ": 值为空");

            size_t used = 0;
            long long x = 0;
            try
            {
                x = std::stoll(v, &used);
            }
            catch (const std::exception&)
            {
                throw std::runtime_error(where + ": 值不是整数 \"" + v + "\"");
            }
            if (used != v.size())
            {
                throw std::runtime_error(where + ": 值不是整数 \"" + v + "\"");
            }
            return static_cast<int64_t>(x);
        }

        // 只认小写的 true / false —— 不做 "1/0/yes/no" 的方言，
        // 少一种写法就少一处以后看不懂的地方。
        bool parse_bool(const std::string& v, const std::string& where)
        {
            if (v == "true")  return true;
            if (v == "false") return false;
            throw std::runtime_error(where + ": 只认 true / false，给的是 \"" + v + "\"");
        }

        // 与 parse_int 同样的严格性：整个字符串都得被吃掉，"0.1x" 不能静默变成 0.1。
        // 另外拒绝科学计数法以外的怪写法不是目的 —— stod 认 1e-4，这个必须能过。
        double parse_double(const std::string& v, const std::string& where)
        {
            if (v.empty()) throw std::runtime_error(where + ": 值为空");

            size_t used = 0;
            double x = 0.0;
            try
            {
                x = std::stod(v, &used);
            }
            catch (const std::exception&)
            {
                throw std::runtime_error(where + ": 值不是数字 \"" + v + "\"");
            }
            if (used != v.size())
            {
                throw std::runtime_error(where + ": 值不是数字 \"" + v + "\"");
            }
            return x;
        }
    }  // namespace

    Settings defaults()
    {
        Settings s;

        s.run.data_dir = "";            // data_dir 故意留空 —— 它必须由 prkd.ini 或命令行给出。
        s.run.out_dir = "./run";        // 相对路径，落点由启动目录决定

        const unsigned hw = std::thread::hardware_concurrency();
        s.loader.workers = (hw > 1) ? static_cast<int64_t>(hw / 2) : 1;

        return s;
    }

    void Settings::load_file(const std::string& path)
    {
        std::ifstream f(path);
        if (!f)
        {
            throw std::runtime_error(
                "打不开配置文件: " + path +
                "（当前目录: " + std::filesystem::current_path().string() + "）");
        }

        std::string line;
        int64_t ln = 0;

        while (std::getline(f, line))
        {
            ++ln;

            // 首行可能带 UTF-8 BOM —— 编辑器很容易加上，先吃掉。
            // 不吃掉的话第一个 key 会变成 "\xEF\xBB\xBFrun.data_dir"，报未知配置项。
            if (ln == 1 && line.size() >= 3 &&
                static_cast<unsigned char>(line[0]) == 0xEF &&
                static_cast<unsigned char>(line[1]) == 0xBB &&
                static_cast<unsigned char>(line[2]) == 0xBF)
            {
                line.erase(0, 3);
            }

            // 行尾注释：路径里出现 '#' 的情况极少，可以接受
            const auto hash = line.find('#');
            if (hash != std::string::npos) line.erase(hash);

            line = trim(line);
            if (line.empty()) continue;

            // 按【第一个】'=' 切分 —— 值里可以再出现 '='
            const auto eq = line.find('=');
            if (eq == std::string::npos)
            {
                throw std::runtime_error(path + ":" + std::to_string(ln) + ": 缺少 '='");
            }

            const std::string key   = trim(line.substr(0, eq));
            const std::string value = trim(line.substr(eq + 1));
            const std::string where = path + ":" + std::to_string(ln);

            if (key.empty()) throw std::runtime_error(where + ": 键名为空");

            // ---- run ----
            if      (key == "run.data_dir")     run.data_dir = value;
            else if (key == "run.out_dir")      run.out_dir = value;
            else if (key == "run.ckpt")         run.ckpt = value;
            else if (key == "run.task")          run.task = value;
            else if (key == "run.dataset")      run.dataset = value;
            else if (key == "run.device_index") run.device_index =
                                                    static_cast<int>(parse_int(value, where));
            else if (key == "run.seed")         run.seed = parse_int(value, where);

            // ---- loader ----
            else if (key == "loader.workers")    loader.workers    = parse_int(value, where);
            else if (key == "loader.batch")      loader.batch      = parse_int(value, where);
            else if (key == "loader.prefetch")   loader.prefetch   = parse_int(value, where);
            else if (key == "loader.short_side") loader.short_side = parse_int(value, where);
            else if (key == "loader.crop")       loader.crop       = parse_int(value, where);
            else if (key == "loader.train")      loader.is_train   = parse_bool(value, where);
            else if (key == "loader.shuffle")    loader.shuffle    = parse_bool(value, where);
            else if (key == "loader.drop_last")  loader.drop_last  = parse_bool(value, where);

            // ---- aloi（只在 run.dataset = aloi 时被读）----
            else if (key == "aloi.view_begin") aloi.view_begin = parse_int(value, where);
            else if (key == "aloi.view_end")   aloi.view_end   = parse_int(value, where);

            // ---- debug ----
            else if (key == "debug.intra")    debug.intra    = static_cast<int>(parse_int(value, where));
            else if (key == "debug.selftest") debug.selftest = parse_bool(value, where);
            else if (key == "debug.log_every") debug.log_every = parse_int(value, where);

            // ---- train（训练超参；不含 batch，批属于 loader）----
            else if (key == "train.epochs")       train.epochs       = parse_int(value, where);
            else if (key == "train.lr")           train.lr           = parse_double(value, where);
            else if (key == "train.momentum")     train.momentum     = parse_double(value, where);
            else if (key == "train.weight_decay") train.weight_decay = parse_double(value, where);
            else if (key == "train.log_every")    train.log_every    = parse_int(value, where);
            else if (key == "train.best_write_from") train.best_write_from = parse_double(value, where);
            else if (key == "train.resume_lr")
            {
                if (value == "config")   train.resume_lr_from_config = true;
                else if (value == "keep") train.resume_lr_from_config = false;
                else throw std::runtime_error(where + ": train.resume_lr 只认 config / keep，给的是 \""
                    + value + "\"");
            }

            // ---- kd（关系蒸馏；只对 task=kd 有意义）----
            else if (key == "kd.keep")        kd.keep        = parse_int(value, where);
            else if (key == "kd.lambda")      kd.lambda      = parse_double(value, where);
            else if (key == "kd.pca_samples") kd.pca_samples = parse_int(value, where);
            else if (key == "kd.pca_source")  kd.pca_source  = value;
            else if (key == "kd.pca_ckpt")    kd.pca_ckpt    = value;

            // ★ 未知 key 一律报错。拼错的 "worker = 8" 静默失效是最坑的一种。
            else throw std::runtime_error(where + ": 未知配置项 \"" + key + "\"");
        }

        // 值域检查集中放在最后：逐行检查的话，多行都错时只会看到第一条
        if (loader.workers    <  0) throw std::runtime_error(path + ": loader.workers 不能为负");
        if (loader.batch      <= 0) throw std::runtime_error(path + ": loader.batch 必须 > 0");
        if (loader.prefetch   <  1) throw std::runtime_error(path + ": loader.prefetch 必须 >= 1");
        if (loader.short_side <= 0) throw std::runtime_error(path + ": loader.short_side 必须 > 0");
        if (loader.crop       <= 0) throw std::runtime_error(path + ": loader.crop 必须 > 0");
        if (debug.intra       <  1) throw std::runtime_error(path + ": debug.intra 必须 >= 1");
        if (debug.log_every   <  0) throw std::runtime_error(path + ": debug.log_every 不能为负（0 = 不打印进度）");
        if (train.epochs      <  1) throw std::runtime_error(path + ": train.epochs 必须 >= 1");
        if (train.lr          <= 0) throw std::runtime_error(path + ": train.lr 必须 > 0");
        if (train.momentum    <  0 || train.momentum >= 1) throw std::runtime_error(path + ": train.momentum 必须在 [0,1) 内");
        if (train.weight_decay < 0) throw std::runtime_error(path + ": train.weight_decay 不能为负");
        if (train.log_every   <  0) throw std::runtime_error(path + ": train.log_every 不能为负（0 = 不打印）");
        if (kd.keep        <  1) throw std::runtime_error(path + ": kd.keep 必须 >= 1（保留的主轴数）");
        if (kd.lambda      <  0) throw std::runtime_error(path + ": kd.lambda 不能为负（0 = 退化成纯 CE）");
        if (kd.pca_samples <  1) throw std::runtime_error(path + ": kd.pca_samples 必须 >= 1");
        if (kd.pca_source != "live" && kd.pca_source != "saved")
            throw std::runtime_error(path + ": kd.pca_source 只认 live / saved，给的是 \"" + kd.pca_source + "\"");
        if (train.best_write_from < 0.0 || train.best_write_from > 1.0) throw std::runtime_error(path + ": train.best_write_from 必须是 [0,1] 的比例（0=每次写，1=只在收尾写）");
        if (run.dataset != "imagenet" && run.dataset != "aloi")
        {
            throw std::runtime_error(path + ": run.dataset 只能是 imagenet 或 aloi，收到 \""+ run.dataset + "\"");
        }
        if (aloi.view_begin < 0 || aloi.view_end > 360 ||
            aloi.view_end <= aloi.view_begin)
        {
            throw std::runtime_error(path + ": aloi 视角范围非法 [" + std::to_string(aloi.view_begin) + ","+ std::to_string(aloi.view_end) + ")，要求 0 <= begin < end <= 360");
        }

    }

    void Settings::print() const
    {
        std::printf("[cfg] ---------- run：在哪训 ----------\n");
        std::printf("[cfg]   task          = %s\n", run.task.c_str());
        std::printf("[cfg]   dataset       = %s\n", run.dataset.c_str());
        std::printf("[cfg]   data_dir      = %s\n", run.data_dir.empty() ? "(未设置)" : run.data_dir.c_str());
        std::printf("[cfg]   out_dir       = %s\n", run.out_dir.c_str());
        std::printf("[cfg]   ckpt          = %s\n", run.ckpt.empty() ? "(不加载)" : run.ckpt.c_str());
        std::printf("[cfg]   device_index  = %d%s\n", run.device_index,run.device_index < 0 ? "（强制 CPU）" : "");
        std::printf("[cfg]   seed          = %lld\n", (long long)run.seed);

        std::printf("[cfg] ---------- loader：怎么读 ----------\n");
        std::printf("[cfg]   workers       = %lld%s\n", (long long)loader.workers,loader.workers <= 0 ? "（同步，不起线程）" : "");
        std::printf("[cfg]   batch         = %lld\n", (long long)loader.batch);
        std::printf("[cfg]   prefetch      = %lld（池深：同时几批在飞）\n", (long long)loader.prefetch);
        std::printf("[cfg]   short_side    = %lld\n", (long long)loader.short_side);
        std::printf("[cfg]   crop          = %lld\n", (long long)loader.crop);

        // 三个开关并列打印：它们互相独立，配出"中心裁剪 + 洗牌 + 丢尾巴"是合法的，
        // 但那多半不是本意 —— 所以让它一眼可见，而不是靠代码拦。
        std::printf("[cfg]   train         = %s（变换：%s）\n",loader.is_train ? "true" : "false",loader.is_train ? "增强" : "中心裁剪");
        std::printf("[cfg]   shuffle       = %s\n", loader.shuffle ? "true" : "false");
        std::printf("[cfg]   drop_last     = %s%s\n", loader.drop_last ? "true" : "false",loader.drop_last ? "" : "（末批可短）");

        // ALOI 的 split 不是目录，是视角范围 —— 单独一节。
        // 只在 dataset=aloi 时打印，免得看 imagenet 的日志时多出一段无关信息。
        if (run.dataset == "aloi")
        {
            std::printf("[cfg] ---------- aloi：视角范围（就是 split）----------\n");
            std::printf("[cfg]   view          = [%lld, %lld)  左闭右开\n",(long long)aloi.view_begin, (long long)aloi.view_end);
        }

        std::printf("[cfg] ---------- train：训练配方（不含 batch，批属于 loader）----------\n");
        std::printf("[cfg]   epochs        = %lld\n", (long long)train.epochs);
        std::printf("[cfg]   lr            = %g\n", train.lr);
        std::printf("[cfg]   momentum      = %g\n", train.momentum);
        std::printf("[cfg]   weight_decay  = %g\n", train.weight_decay);
        std::printf("[cfg]   log_every     = %lld%s\n", (long long)train.log_every,train.log_every == 0 ? "（不打印）" : " 批");
        std::printf("[cfg]   resume_lr     = %s（续跑时 lr 取自 %s）\n",train.resume_lr_from_config ? "config" : "keep",train.resume_lr_from_config ? "本配置" : "opt.pt");
        std::printf("[cfg]   best_write_from = %.2f（best 从训练后 %.0f%% 处开始写盘，之前只在内存）\n",train.best_write_from, (1.0 - train.best_write_from) * 100.0);

        std::printf("[cfg] ---------- kd：关系蒸馏（只在 task=kd 时读）----------\n");
        std::printf("[cfg]   keep          = %lld（PCA 保留的主轴数）\n", (long long)kd.keep);
        std::printf("[cfg]   lambda        = %.2f（关系损失权重；0 = 退化成纯 CE）\n", kd.lambda);
        std::printf("[cfg]   pca_samples   = %lld（拟合 PCA 用前多少张训练图）\n", (long long)kd.pca_samples);
        std::printf("[cfg]   pca_ckpt      = %s\n", kd.pca_ckpt.empty()
            ? "（空 —— 用训练中的学生）" : kd.pca_ckpt.c_str());
        std::printf("[cfg]   pca_source    = %s（%s）\n", kd.pca_source.c_str(),
            kd.pca_source == "saved" ? "拟合一次后存盘，之后读回同一把尺子" : "每次启动在当前学生上重拟合");

        std::printf("[cfg] ---------- debug：诊断 ----------\n");
        std::printf("[cfg]   intra         = %d\n", debug.intra);
        std::printf("[cfg]   selftest      = %s\n", debug.selftest ? "true" : "false");
        std::printf("[cfg]   log_every     = %lld%s\n", (long long)debug.log_every,debug.log_every == 0 ? "（不打印进度）" : " 批");
    }

    std::string Settings::summary() const
    {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "dataset=%s batch=%lld lr=%g epochs=%lld seed=%lld device=%d",
            run.dataset.c_str(),
            (long long)loader.batch,
            train.lr,
            (long long)train.epochs,
            (long long)run.seed,
            run.device_index);
        return std::string(buf);
    }
}  // namespace prkd::cfg

