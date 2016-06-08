

#-----------------------------------------------------------#
#															#
#				GPU Meshing Library V 1.00					#
#															#
#-----------------------------------------------------------#
#															#
#	Description:		Easy mesh programing with OpenCl	#
#	Author:				Loic MARECHAL						#
#	Creation date:		jul 02 2010							#
#	Last modification:	jul 05 2010							#
#															#
#-----------------------------------------------------------#


CC     = gcc
CFLAGS = -O -s


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
LIB = gmlib1.a


# Definition of the compiling implicit rule

$(OBJSDIR)/%.o : $(SRCSDIR)/%.c
	$(CC) -c -O -I$(SRCSDIR) $< -o $@


# Install the library

$(LIBDIR)/$(LIB): $(DIRS) $(OBJS)
	cp $(OBJSDIR)/gmlib1.o $@
	cp $(SRCSDIR)/*.h $(INCDIR)


# Objects depends on headers

$(OBJS): $(HDRS)


# Build the working directories

$(DIRS):
	@[ -d $@ ] || mkdir $@


# Remove temporary files

clean:
	rm -f $(OBJS) $(LIBDIR)/$(LIB)

# Build a dated archive including sources, patterns and makefile

tar: $(DIRS)
	tar czf $(ARCHDIR)/gmlib1.`date +"%Y.%m.%d"`.tgz sources tests Makefile

zip: $(DIRS)
	archive_gm1.sh
