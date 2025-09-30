(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        blue - block
        yellow - block
        red - block
        green - block
        orange - block
        purple - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable blue)
        (on yellow blue)
        (on red yellow)
        (on green red)
        (clear green)
        (ontable orange)
        (on purple orange)
        (clear purple)
        ; blue -> yellow -> red -> green
        ; orange -> purple
    )
    (:goal (and (on orange green)
        (on purple yellow)
        (on blue purple)
        (on red blue)))
)
    