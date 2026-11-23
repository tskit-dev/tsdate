project('tsdate', 'c', default_options: ['c_std=c11'])

cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required : false)
cunit_dep = dependency('cunit')

extra_c_args = [
    '-Wall', '-Wextra', '-Werror', '-Wpedantic', '-W',
    '-Wmissing-prototypes',  '-Wstrict-prototypes',
    '-Wconversion', '-Wshadow', '-Wpointer-arith', '-Wcast-align',
    '-Wcast-qual', '-Wwrite-strings', '-Wnested-externs',
    '-fshort-enums', '-fno-common', '-ffp-contract=off']

if get_option('buildtype') == 'debug'
    extra_c_args += ['-DTSD_TRACE_ERRORS']
endif

tsdate_sources = ['err.c', 'hypergeo.c', 'approx.c', 'rescaling.c',
    'phasing.c', 'util.c', 'prior.c']

tsdate_lib = static_library('tsdate',
    sources: tsdate_sources, dependencies: [m_dep],
    c_args: extra_c_args)

unit_tests = executable('tests',
    sources: ['tests/tests.c'],
    link_with: [tsdate_lib], dependencies: [cunit_dep, m_dep])
test('Unit tests', unit_tests)
