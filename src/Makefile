# 定义 CUDA 编译器
NVCC = nvcc

# 检测系统类型（Windows/MSYS or Linux/Unix）
UNAME_S := $(shell uname -s)

ifeq ($(findstring MINGW,$(UNAME_S)),MINGW)  # Git Bash / Windows MSYS
    EXT = dll
    RM = del /Q
else
    EXT = so
    RM = rm -f
endif

# 定义源文件列表（不带扩展名）
SRCNAMES = pml_updates_e pml_updates_h back uc fields_updates_gpu sourcereceiver

# 构造目标文件路径
TARGETS = $(addprefix lib/,$(addsuffix .$(EXT),$(SRCNAMES)))

# 默认目标
all: $(TARGETS)

# 规则：编译 lib/ 目录中的 .cu 文件
lib/%.$(EXT): lib/%.cu
	$(NVCC) -shared -Xcompiler -fPIC -o $@ $< -D_GLIBCXX_USE_CXX11_ABI=0

# 清理生成文件
clean:
	$(RM) $(TARGETS)
