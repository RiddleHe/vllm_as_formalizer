(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        green - block
        blue - block
        orange - block
        red - block
        yellow - block
        purple - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable green)
        (on blue green)
        (on orange blue)
        (on red orange)
        (on yellow red)
        (on purple yellow)
        (clear purple)
        ; green -> blue -> orange -> red -> yellow -> purple
    )
    (:goal (and (on purple red)
        (on green blue)
        (on orange green)
        (on yellow orange)))
)
    