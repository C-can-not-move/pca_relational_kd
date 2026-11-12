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

    void load_file(const std::string& path, Settings& s)
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
            if      (key == "run.data_dir")     s.run.data_dir = value;
            else if (key == "run.out_dir")      s.run.out_dir = value;
            else if (key == "run.ckpt")         s.run.ckpt = value;
            else if (key == "run.dataset")      s.run.dataset = value;
            else if (key == "run.device_index") s.run.device_index =
                                                    static_cast<int>(parse_int(value, where));
            else if (key == "run.seed")         s.run.seed = parse_int(value, where);

            // ---- loader ----
            else if (key == "loader.workers")    s.loader.workers    = parse_int(value, where);
            else if (key == "loader.batch")      s.loader.batch      = parse_int(value, where);
            else if (key == "loader.prefetch")   s.loader.prefetch   = parse_int(value, where);
            else if (key == "loader.short_side") s.loader.short_side = parse_int(value, where);
            else if (key == "loader.crop")       s.loader.crop       = parse_int(value, where);
            else if (key == "loader.train")      s.loader.is_train   = parse_bool(value, where);
            else if (key == "loader.shuffle")    s.loader.shuffle    = parse_bool(value, where);
            else if (key == "loader.drop_last")  s.loader.drop_last  = parse_bool(value, where);

            // ---- aloi（只在 run.dataset = aloi 时被读）----
            else if (key == "aloi.view_begin") s.aloi.view_begin = parse_int(value, where);
            else if (key == "aloi.view_end")   s.aloi.view_end   = parse_int(value, where);

            // ---- debug ----
            else if (key == "debug.intra")    s.debug.intra    = static_cast<int>(parse_int(value, where));
            else if (key == "debug.selftest") s.debug.selftest = parse_bool(value, where);
            else if (key == "debug.log_every") s.debug.log_every = parse_int(value, where);

            // ★ 未知 key 一律报错。拼错的 "worker = 8" 静默失效是最坑的一种。
            else throw std::runtime_error(where + ": 未知配置项 \"" + key + "\"");
        }

        // 值域检查集中放在最后：逐行检查的话，多行都错时只会看到第一条
        if (s.loader.workers    <  0) throw std::runtime_error(path + ": loader.workers 不能为负");
        if (s.loader.batch      <= 0) throw std::runtime_error(path + ": loader.batch 必须 > 0");
        if (s.loader.prefetch   <  1) throw std::runtime_error(path + ": loader.prefetch 必须 >= 1");
        if (s.loader.short_side <= 0) throw std::runtime_error(path + ": loader.short_side 必须 > 0");
        if (s.loader.crop       <= 0) throw std::runtime_error(path + ": loader.crop 必须 > 0");
        if (s.debug.intra       <  1) throw std::runtime_error(path + ": debug.intra 必须 >= 1");
        if (s.debug.log_every   <  0) throw std::runtime_error(path + ": debug.log_every 不能为负（0 = 不打印进度）");
        if (s.run.dataset != "imagenet" && s.run.dataset != "aloi")
        {
            throw std::runtime_error(path + ": run.dataset 只能是 imagenet 或 aloi，收到 \""
                                     + s.run.dataset + "\"");
        }
        if (s.aloi.view_begin < 0 || s.aloi.view_end > 360 ||
            s.aloi.view_end <= s.aloi.view_begin)
        {
            throw std::runtime_error(path + ": aloi 视角范围非法 ["
                                     + std::to_string(s.aloi.view_begin) + ","
                                     + std::to_string(s.aloi.view_end) + ")，要求 0 <= begin < end <= 360");
        }

    }

    void print(const Settings& s)
    {
        std::printf("[cfg] ---------- run：在哪训 ----------\n");
        std::printf("[cfg]   dataset       = %s\n", s.run.dataset.c_str());
        std::printf("[cfg]   data_dir      = %s\n", s.run.data_dir.empty() ? "(未设置)" : s.run.data_dir.c_str());
        std::printf("[cfg]   out_dir       = %s\n", s.run.out_dir.c_str());
        std::printf("[cfg]   ckpt          = %s\n", s.run.ckpt.empty() ? "(不加载)" : s.run.ckpt.c_str());
        std::printf("[cfg]   device_index  = %d%s\n", s.run.device_index,
                    s.run.device_index < 0 ? "（强制 CPU）" : "");
        std::printf("[cfg]   seed          = %lld\n", (long long)s.run.seed);

        std::printf("[cfg] ---------- loader：怎么读 ----------\n");
        std::printf("[cfg]   workers       = %lld%s\n", (long long)s.loader.workers,
                    s.loader.workers <= 0 ? "（同步，不起线程）" : "");
        std::printf("[cfg]   batch         = %lld\n", (long long)s.loader.batch);
        std::printf("[cfg]   prefetch      = %lld（池深：同时几批在飞）\n", (long long)s.loader.prefetch);
        std::printf("[cfg]   short_side    = %lld\n", (long long)s.loader.short_side);
        std::printf("[cfg]   crop          = %lld\n", (long long)s.loader.crop);

        // 三个开关并列打印：它们互相独立，配出"中心裁剪 + 洗牌 + 丢尾巴"是合法的，
        // 但那多半不是本意 —— 所以让它一眼可见，而不是靠代码拦。
        std::printf("[cfg]   train         = %s（变换：%s）\n",
                    s.loader.is_train ? "true" : "false",
                    s.loader.is_train ? "增强" : "中心裁剪");
        std::printf("[cfg]   shuffle       = %s\n", s.loader.shuffle ? "true" : "false");
        std::printf("[cfg]   drop_last     = %s%s\n", s.loader.drop_last ? "true" : "false",
                    s.loader.drop_last ? "" : "（末批可短）");

        // ALOI 的 split 不是目录，是视角范围 —— 单独一节。
        // 只在 dataset=aloi 时打印，免得看 imagenet 的日志时多出一段无关信息。
        if (s.run.dataset == "aloi")
        {
            std::printf("[cfg] ---------- aloi：视角范围（就是 split）----------\n");
            std::printf("[cfg]   view          = [%lld, %lld)  左闭右开\n",
                        (long long)s.aloi.view_begin, (long long)s.aloi.view_end);
        }

        std::printf("[cfg] ---------- debug：诊断 ----------\n");
        std::printf("[cfg]   intra         = %d\n", s.debug.intra);
        std::printf("[cfg]   selftest      = %s\n", s.debug.selftest ? "true" : "false");
        std::printf("[cfg]   log_every     = %lld%s\n", (long long)s.debug.log_every,
                    s.debug.log_every == 0 ? "（不打印进度）" : " 批");
    }
}  // namespace prkd::cfg
