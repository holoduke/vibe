#!/usr/bin/perl
# Split line ranges of a GLSL file into #include files (textual, order kept).
# usage: split_shader.pl FILE start-end:name.glsl:"one-line description" ...
# A range start is moved up over the comment lines directly above it.
use strict; my $f = shift; open F,"<:raw",$f or die; my @l = <F>; close F;
my @specs = map { my ($r,$n,$d) = split /:/, $_, 3; my ($a,$b) = split /-/, $r; [$a,$b,$n,$d] } @ARGV;
@specs = sort { $a->[0] <=> $b->[0] } @specs;
for my $s (@specs) { while ($s->[0] > 1 && $l[$s->[0]-2] =~ /^\s*\/\//) { $s->[0]--; } }
for my $i (0..$#specs-1) { $specs[$i][1] = $specs[$i+1][0]-1 if $specs[$i][1] >= $specs[$i+1][0]; }
(my $dir = $f) =~ s/[^\/]*$//; (my $base = $f) =~ s/.*\///;
for my $s (reverse @specs) {
  my ($a,$b,$n,$d) = @$s;
  my @chunk = splice(@l, $a-1, $b-$a+1, "#include \"$n\"   // $d\n");
  open O,">:raw","$dir$n" or die; print O "// $n - part of $base (included there, in place; not a standalone shader).\n// $d\n\n", @chunk; close O;
  print "$n: ".scalar(@chunk)." lines\n";
}
open F,">:raw",$f; print F @l; close F; print "$f: ".scalar(@l)." lines left\n";
