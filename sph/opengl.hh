#ifndef OPENGL_HH
#define OPENGL_HH

#include <GL/glew.h>

#include "theora.hh"

int window_width = 800, window_height = 600;
GLuint fbo, render_buf;
bool no_screen = false;
thx::screen_recorder recorder("out.ogv", window_width, window_height);

void init_opengl(float point_size) {
    if (no_screen) {
        // https://stackoverflow.com/questions/12157646/how-to-render-offscreen-on-opengl
        glGenRenderbuffers(1,&render_buf);
        glBindRenderbuffer(GL_RENDERBUFFER, render_buf);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, window_width, window_height);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        glGenFramebuffers(1,&fbo);
        glBindFramebuffer(GL_FRAMEBUFFER,fbo);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_RENDERBUFFER, render_buf);
    }
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glPointSize(point_size);
    glClearColor(0.9f,0.9f,0.9f,1);
    glMatrixMode(GL_PROJECTION);
    //glDeleteFramebuffers(1,&fbo);
    //glDeleteRenderbuffers(1,&render_buf);
}

void on_reshape(GLint new_width, GLint new_height) {
    if (no_screen) { glBindFramebuffer(GL_FRAMEBUFFER,fbo); }
    window_width = new_width, window_height = new_height;
    glViewport(0, 0, window_width, window_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, window_width, 0, window_height);
    if (no_screen) { glBindFramebuffer(GL_FRAMEBUFFER,0); }
}

#endif // vim:filetype=cpp
