

#-----------------------------------------------------------#
#															#
#				GPU Meshing Library V 2.00					#
#															#
#-----------------------------------------------------------#
#															#
#	Description:		Easy mesh programing with OpenCl	#
#	Author:				Loic MARECHAL						#
#	Creation date:		jul 02 2010							#
#	Last modification:	nov 26 2012							#
#															#
#-----------------------------------------------------------#


CFLAGS = -O

# Working directories
LIBDIR  = $(HOME)/lib/$(ARCHI)
INCDIR  = $(HOME)/include
SRCSDIR = sources
OBJSDIR = objects/$(ARCHI)
ARCHDIR = archives
DIRS    = objects $(LIBDIR) $(OBJSDIR) $(ARCHDIR) $(INCDIR)
VPATH   = $(SRCSDIR)


# Files to be compiled
SRCS = $(wildcard $(SRCSDIR)/*.c)
HDRS = $(wildcard $(SRCSDIR)/*.h)
OBJS = $(patsubst $(SRCSDIR)%, $(OBJSDIR)%, $(SRCS:.c=.o))
OCLS = $(wildcard $(SRCSDIR)/*.cl)
OCLH = $(patsubst $(SRCSDIR)%, $(SRCSDIR)%, $(OCLS:.cl=.h))
LIB = gmlib2.a


# Define OpenCL compiling implicit rules
$(SRCSDIR)/%.h : $(SRCSDIR)/%.cl
	cl2h $< $@ `basename $@ .h`

# Definition of the compiling implicit rule
$(OBJSDIR)/%.o : $(SRCSDIR)/%.c
	$(CC) -c -O -I$(SRCSDIR) $< -o $@

# Install the library
$(LIBDIR)/$(LIB): $(DIRS) $(OCLH) $(OBJS)
	cp $(OBJSDIR)/gmlib2.o $@
	cp $(SRCSDIR)/*.h $(INCDIR)


# OpenCL headers depend on OpenCL sources .cl
$(OCLH): $(OCLS)

# Objects depends on headers
$(OBJS): $(HDRS)


# Build the working directories
$(DIRS):
	@[ -d $@ ] || mkdir $@


# Remove temporary files
clean:
	rm -f $(OBJS) $(LIBDIR)/$(LIB) $(OCLH) utils/cl2h

# Build a dated archive including sources, patterns and makefile
tar: $(DIRS)
	tar czf $(ARCHDIR)/gmlib2.`date +"%Y.%m.%d"`.tgz sources utils Makefile

dist: $(DIRS)
	tar czf ~/Desktop/gmlib2.`date +"%Y.%m.%d"`.tgz sources examples utils Makefile LICENSE_lgpl.txt copyright.txt
