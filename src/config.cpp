#include "config.h"

namespace prkd::cfg
{
	RunConfig defaults()
	{
		RunConfig c;
#ifdef _WIN32
		c.data_dir = "D:/DATA/CODE/C++/pca_relational_kd/dataset/tiny-imagenet-200";
		c.out_dir = "D:/DATA/CODE/C++/pca_relational_kd/run";
#else
		c.data_dir = "/home/chenzhijiang/userFile/dataset_all/tiny-imagenet-200";
#endif
		return c;
	}
}  // namespace prkd::cfg