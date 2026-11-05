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

            // ---- debug ----
            else if (key == "debug.intra")    s.debug.intra    = static_cast<int>(parse_int(value, where));
            else if (key == "debug.selftest") s.debug.selftest = parse_bool(value, where);

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

    }

    void print(const Settings& s)
    {
        std::printf("[cfg] ---------- run：在哪训 ----------\n");
        std::printf("[cfg]   data_dir      = %s\n", s.run.data_dir.empty() ? "(未设置)" : s.run.data_dir.c_str());
        std::printf("[cfg]   out_dir       = %s\n", s.run.out_dir.c_str());
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
        std::printf("[cfg]   train         = %s%s\n", s.loader.is_train ? "true" : "false",
                    s.loader.is_train ? "（洗牌 + drop_last）" : "（不洗牌，末批可短）");

        std::printf("[cfg] ---------- debug：诊断 ----------\n");
        std::printf("[cfg]   intra         = %d\n", s.debug.intra);
        std::printf("[cfg]   selftest      = %s\n", s.debug.selftest ? "true" : "false");
    }
}  // namespace prkd::cfg
