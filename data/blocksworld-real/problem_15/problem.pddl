(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        orange - block
        purple - block
        blue - block
        red - block
        yellow - block
        green - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable orange)
        (on purple orange)
        (on blue purple)
        (on red blue)
        (on yellow red)
        (on green yellow)
        (clear green)
        ; orange -> purple -> blue -> red -> yellow -> green
    )
    (:goal (and (on green red)
        (on orange green)
        (on purple orange)
        (on blue purple)
        (on yellow blue)))
)
    