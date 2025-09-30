(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        blue - block
        purple - block
        orange - block
        red - block
        green - block
        yellow - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable blue)
        (on purple blue)
        (on orange purple)
        (on red orange)
        (on green red)
        (on yellow green)
        (clear yellow)
        ; blue -> purple -> orange -> red -> green -> yellow
    )
    (:goal (and (on purple yellow)
        (on green red)
        (on blue orange)))
)
    