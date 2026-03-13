/*******************************************************************
 *       Filename:  tomo.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  06/15/2020 05:48:48 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:
 *          Email:
 *        Company:
 *
 *******************************************************************/
#include <stdio.h>
#include "util.h"
#include "mrc.h"
#include "CTFAlgo.h"
#include "ReconstructionAlgo_WBP_RAM.h"
#include "time.h"
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <chrono>
#include <vector>

static bool isTrueValue(const string& v)
{
    return (v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "YES" || v == "on" || v == "ON");
}

static bool getBoolPara(const map<string, string>& para, const string& key, bool defv)
{
    map<string, string>::const_iterator it = para.find(key);
    if (it == para.end()) return defv;
    return isTrueValue(it->second);
}

static string getStringPara(const map<string, string>& para, const string& key, const string& defv)
{
    map<string, string>::const_iterator it = para.find(key);
    if (it == para.end()) return defv;
    return it->second;
}

static bool isAbsPath(const string& p)
{
    return (!p.empty() && p[0] == '/');
}

static string joinPath(const string& dir, const string& name)
{
    if (name.empty()) return dir;
    if (isAbsPath(name)) return name;
    if (dir.empty()) return name;
    if (dir[dir.size() - 1] == '/') return dir + name;
    return dir + "/" + name;
}

static string dirnameOf(const string& path)
{
    size_t pos = path.find_last_of('/');
    if (pos == string::npos) return ".";
    if (pos == 0) return "/";
    return path.substr(0, pos);
}

static bool ensureDirRecursive(const string& dir)
{
    if (dir.empty() || dir == ".") return true;
    if (dir == "/") return true;

    string cur;
    size_t i = 0;
    if (dir[0] == '/')
    {
        cur = "/";
        i = 1;
    }

    while (i < dir.size())
    {
        while (i < dir.size() && dir[i] == '/') i++;
        if (i >= dir.size()) break;
        size_t j = i;
        while (j < dir.size() && dir[j] != '/') j++;

        string token = dir.substr(i, j - i);
        if (!token.empty() && token != ".")
        {
            if (cur.empty() || cur == "/")
                cur = (cur == "/") ? (cur + token) : token;
            else
                cur += "/" + token;

            if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST)
            {
                fprintf(stderr, "[STAGE] mkdir failed for [%s], errno=%d\n", cur.c_str(), errno);
                return false;
            }
        }
        i = j;
    }

    return true;
}

static bool copyFileBuffered(const string& src, const string& dst, double& ms_out)
{
    ms_out = 0.0;
    if (src == dst) return true;

    auto t_begin = std::chrono::steady_clock::now();

    FILE* fin = fopen(src.c_str(), "rb");
    if (!fin)
    {
        fprintf(stderr, "[STAGE] open src failed: %s\n", src.c_str());
        return false;
    }

    if (!ensureDirRecursive(dirnameOf(dst)))
    {
        fclose(fin);
        return false;
    }

    FILE* fout = fopen(dst.c_str(), "wb");
    if (!fout)
    {
        fprintf(stderr, "[STAGE] open dst failed: %s\n", dst.c_str());
        fclose(fin);
        return false;
    }

    const size_t kBuf = 8ull * 1024ull * 1024ull;
    std::vector<char> buf(kBuf);

    bool ok = true;
    while (true)
    {
        size_t nr = fread(buf.data(), 1, buf.size(), fin);
        if (nr > 0)
        {
            size_t nw = fwrite(buf.data(), 1, nr, fout);
            if (nw != nr)
            {
                ok = false;
                break;
            }
        }

        if (nr < buf.size())
        {
            if (feof(fin)) break;
            ok = false;
            break;
        }
    }

    fflush(fout);
    fclose(fout);
    fclose(fin);

    auto t_end = std::chrono::steady_clock::now();
    ms_out = std::chrono::duration<double, std::milli>(t_end - t_begin).count();
    return ok;
}

inline void getTime(struct timeval start_time)
{
    struct timeval end_time;
	gettimeofday(&end_time, NULL);

	double diff = 1000000 * (end_time.tv_sec-start_time.tv_sec)+ end_time.tv_usec-start_time.tv_usec;

	printf(" time = %lf us\n", diff);
	/*
    double seconds_total=difftime(end_time,start_time);
    int hours=((int)seconds_total)/3600;
    int minutes=(((int)seconds_total)%3600)/60;
    int seconds=(((int)seconds_total)%3600)%60;

    cout << "Time elapsed: ";
    if(hours>0)
    {
        cout << hours << "h ";
    }
    if(minutes > 0 || hours > 0)
    {
        cout << minutes << "m ";
    }*/
    //cout << seconds << "s" << endl << endl;
    //

}

int main(int argc, char **argv)
{

    if(argc != 2)
    {
        fprintf(stderr, "\n  Usage: \n%6s%s /path/to/parameter/file\n\n", "", argv[0]);
        exit(1);
    }


	struct timeval start_time;
    gettimeofday(&start_time, NULL);

    //time_t start_time;
    //time(&start_time);

    map<string, string> inputPara;
    map<string, string> outputPara;
    const char *paraFileName = argv[1];
    readParaFile(inputPara, paraFileName);
    getAllParas(inputPara);

    // Optional local-disk input staging:
    // copy input files to local disk, then run reconstruction with output kept on original path.
    // This keeps copy-in + compute + final write in one end-to-end timer.
    bool stage_enable = getBoolPara(inputPara, "local_stage_enable", false);
    string stage_root = getStringPara(inputPara, "local_stage_dir", "");

    if (const char* env_stage_enable = getenv("WBP_LOCAL_STAGE_ENABLE"))
    {
        stage_enable = (atoi(env_stage_enable) != 0);
    }
    if (const char* env_stage_root = getenv("WBP_LOCAL_STAGE_DIR"))
    {
        stage_root = string(env_stage_root);
        if (!stage_root.empty()) stage_enable = true;
    }
    string stage_orig_path = getStringPara(inputPara, "path", "./");
    string stage_path;
    double stage_copy_in_ms_total = 0.0;

    if (stage_enable)
    {
        if (stage_root.empty()) stage_root = "/tmp";
        long long ts = (long long)time(NULL);
        int pid = (int)getpid();
        stage_path = joinPath(stage_root, string("wbp_stage_") + std::to_string((long long)pid) + "_" + std::to_string(ts));

        if (!ensureDirRecursive(stage_path))
        {
            const char* user_env = getenv("USER");
            string user_name = (user_env && user_env[0] != '\0') ? string(user_env) : string("user");
            string fallback_root = joinPath(joinPath("/tmp", user_name), "wbp_stage");
            string fallback_path = joinPath(fallback_root, string("wbp_stage_") + std::to_string((long long)pid) + "_" + std::to_string(ts));

            fprintf(stderr, "[STAGE] failed to create local stage dir: %s\n", stage_path.c_str());
            fprintf(stderr, "[STAGE] fallback to: %s\n", fallback_path.c_str());

            stage_root = fallback_root;
            stage_path = fallback_path;
            if (!ensureDirRecursive(stage_path))
            {
                fprintf(stderr, "[STAGE] failed to create fallback local stage dir: %s\n", stage_path.c_str());
                exit(2);
            }
        }

        printf("[STAGE] enabled=1\n");
        printf("[STAGE] original path=%s\n", stage_orig_path.c_str());
        printf("[STAGE] local path=%s\n", stage_path.c_str());

        const char* k_inputs[] = {"input_mrc", "input_tlt", "defocus_file"};
        for (size_t i = 0; i < sizeof(k_inputs) / sizeof(k_inputs[0]); ++i)
        {
            string key = k_inputs[i];
            map<string, string>::iterator it = inputPara.find(key);
            if (it == inputPara.end() || it->second.empty()) continue;

            string src = joinPath(stage_orig_path, it->second);
            string dst = joinPath(stage_path, it->second);
            double one_ms = 0.0;
            if (!copyFileBuffered(src, dst, one_ms))
            {
                fprintf(stderr, "[STAGE] copy-in failed for key=%s src=%s dst=%s\n",
                        key.c_str(), src.c_str(), dst.c_str());
                exit(2);
            }
            // Keep output path unchanged; remap only the input files to local staged copies.
            inputPara[key] = dst;
            stage_copy_in_ms_total += one_ms;
            printf("[STAGE] copy-in %s : %.3f ms\n", key.c_str(), one_ms);
        }
        printf("[STAGE] copy-in total=%.3f ms\n", stage_copy_in_ms_total);
    }

    bool do_alignment=0,do_CTF=0,do_reconstruction_WBP=0,do_reconstruction_WBP_in_RAM=0,do_reconstruction_SIRT=0,do_reconstruction_SIRT_in_RAM=0,do_reconstruction_FD=0;
    map<string,string>::iterator it=inputPara.find("do_alignment");
    if(it!=inputPara.end())
    {
        do_alignment=atoi(it->second.c_str());
    }
    it=inputPara.find("do_CTF");
    if(it!=inputPara.end())
    {
        do_CTF=atoi(it->second.c_str());
    }
    
    it=inputPara.find("do_reconstruction_WBP_in_RAM");
    if(it!=inputPara.end())
    {
        do_reconstruction_WBP_in_RAM=atoi(it->second.c_str());
    }

    if(do_CTF)
    {
	printf("\nDo CTF!\n\n");
        //cout << endl << "Do CTF!" << endl << endl;
        CTFBase *cb = new CTFAlgo();
        cb->doCTF(inputPara, outputPara);
        delete cb;
    }
     
    if(do_reconstruction_WBP_in_RAM)
    {
		printf("\nDo Reconstruction with WBP in RAM!\n\n");
        //cout << endl << "Do Reconstruction with WBP in RAM!" << endl << endl;
        ReconstructionBase *rb = new ReconstructionAlgo_WBP_RAM();
        rb->doReconstruction(inputPara, outputPara);
        delete rb;
    }

	printf("\nFinish!\n");
    //cout << endl << "Finish!" << endl;
    getTime(start_time);

    return 0;
}

